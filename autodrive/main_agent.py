# -*- coding: utf-8 -*-
"""
自包含入口：抓屏 → 感知(Lab自适应) → 姿态(ey/epsi) → 控制(Stanley/PurePursuit) → vgamepad
依赖：opencv-python, numpy, mss, vgamepad, pyyaml, pywin32(可选)
运行：
  python -m autodrive.main_agent autodrive/configs/base_stanley.yaml
热键：
  p  切换是否下发到手柄
  s  保存当前抓屏到当前目录
  q  退出
"""
import sys, time, math, threading
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from autodrive.tools.pose_debug import set_viewport
import numpy as np
import cv2
import mss
import yaml

# 可选：仅用于“只有当前窗口标题包含 Forza 才下发”
try:
    import win32gui
except Exception:
    win32gui = None

# ==== 引入你已放入项目的 Agent 和 Controller ====
from autodrive.agents.base_agent import BaseAgent, SpeedCtrlCfg
# （controllers 已由 BaseAgent 内部按 type 加载）

# ==== 手柄下发（vgamepad） ====
from vgamepad import VX360Gamepad

def _flt_to_i16(x: float) -> int:
    x = max(-1.0, min(1.0, x))
    return int(x * 32767)

class GamepadAct:
    def __init__(self):
        self.pad = VX360Gamepad()
    def send(self, steer: float, throttle: float, brake: float):
        self.pad.left_joystick(x_value=_flt_to_i16(steer), y_value=0)
        self.pad.right_trigger(value=int(max(0.0, min(1.0, throttle)) * 255))
        self.pad.left_trigger(value=int(max(0.0, min(1.0, brake)) * 255))
        self.pad.update()

# ==== 配置 ====
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==== 抓屏 ====
class ScreenGrabber:
    def __init__(self, mon: Dict[str, int]):
        self.mon = mon
        self.sct = mss.mss()
    def grab(self) -> np.ndarray:
        raw = np.array(self.sct.grab(self.mon))[:, :, :3]  # BGRA->BGR
        return raw

# ==== 感知（Lab自适应 ΔE） ====
class LabRoadSeg:
    def __init__(self, roi_y0: int, roi_y1: int, lab_window=(0.30, 0.18), lab_delta: float = 16.0):
        self.y0, self.y1 = int(roi_y0), int(roi_y1)
        self.win_w_ratio, self.win_h_ratio = lab_window
        self.lab_delta = float(lab_delta)

    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        H, W = bgr.shape[:2]
        y0, y1 = np.clip(self.y0, 0, H-1), np.clip(self.y1, 0, H)
        roi = bgr[y0:y1, :, :]
        h, w = roi.shape[:2]
        ww = max(4, int(w * self.win_w_ratio))
        hh = max(4, int(h * self.win_h_ratio))
        x0 = (w - ww) // 2
        yb = h - hh - 4

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        patch = roi[yb:yb+hh, x0:x0+ww]
        mean_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1,3).mean(0)

        de = np.sqrt(np.sum((roi_lab - mean_lab.reshape(1,1,3))**2, axis=2))
        mask = (de < self.lab_delta).astype(np.uint8) * 255

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 仅保留与底边相连的区域
        bottom = mask[-1,:] > 0
        if bottom.any():
            vis = np.zeros_like(mask)
            h2, w2 = mask.shape
            step = max(1, w2//50)
            for x in np.where(bottom)[0][::step]:
                cv2.floodFill(mask, None, (int(x), h2-2), 127)
            vis[mask==127] = 255
            mask = vis

        return mask, (0, y0, W, y1 - y0)

# ==== 姿态估计（ey/epsi） ====
@dataclass
class Pose:
    ey: float      # 横向误差(米，+左)
    epsi: float    # 航向误差(弧度，+逆时针)
    v: float       # 速度(这里先用固定目标，或之后接OBD)

class PoseEstimator:
    def __init__(self, pix2m: float = 0.02, min_area: int = 1500):
        self.p2m = float(pix2m)
        self.min_area = int(min_area)
        self.v_nominal = 10.0  # 先用 nominal 速度（m/s）

    def estimate(self, mask: np.ndarray, roi: Tuple[int,int,int,int]) -> Pose:
        h, w = mask.shape
        M = cv2.moments(mask)
        center_x = w/2.0
        if M["m00"] > self.min_area:
            cx = M["m10"]/M["m00"]
            ey_pix = cx - center_x
            ey = float(ey_pix * self.p2m)
        else:
            ey = 0.0

        # PCA 估朝向（epsi）
        epsi = 0.0
        ys, xs = np.where(mask > 0)
        if xs.size > self.min_area//10:
            pts = np.column_stack((xs, ys)).astype(np.float32)
            mean, eigvecs = cv2.PCACompute(pts, mean=None, maxComponents=2)
            # 主方向向量
            vx, vy = eigvecs[0]
            # 相机坐标系下，x 右 y 下；车辆前向 ~ -y
            # 计算与竖直向上的夹角（负y）
            ang = math.atan2(vy, vx)  # 与x轴夹角
            # 将其转成与 -y 方向的夹角
            desired = -math.pi/2
            epsi = (ang - desired)
            # wrap to [-pi, pi]
            epsi = math.atan2(math.sin(epsi), math.cos(epsi))

        return Pose(ey=ey, epsi=epsi, v=self.v_nominal)

# ==== 工具 ====
def is_foreground_forza() -> bool:
    if win32gui is None:
        return True
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd) or ""
    return ("Forza" in title) or ("Horizon" in title)

# ==== 主程序 ====
def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    mon = cfg["env"]["monitor"]
    y0, y1 = cfg["env"]["roi"]["y0"], cfg["env"]["roi"]["y1"]
    visualize = bool(cfg["debug"]["visualize"])
    send_actuation = bool(cfg["debug"]["actuate"])
    hz = int(cfg["loop"]["hz"])

    print("[debug] env.monitor:", mon)
    print("[debug] env.roi:", cfg["env"]["roi"])
    set_viewport(mon, cfg["env"]["roi"])

    grabber = ScreenGrabber(mon)
    seg = LabRoadSeg(y0, y1,
                     lab_window=tuple(cfg["perception"]["lab_window"]),
                     lab_delta=float(cfg["perception"]["lab_delta"]))
    pose_est = PoseEstimator(pix2m=float(cfg["state"]["pix2meter"]),
                             min_area=int(cfg["state"]["min_area"]))

    # Agent & speed
    ctl_type = cfg.get("controller", {}).get("type", "stanley")
    ctl_params = {k:v for k,v in cfg.get("controller", {}).items() if k != "type"}
    spd_cfg = SpeedCtrlCfg(vmax=float(cfg.get("planner", {}).get("vmax", 12.0)))
    agent = BaseAgent(controller_type=ctl_type, ctl_params=ctl_params, speed_cfg=spd_cfg)

    act = GamepadAct()
    dt = 1.0/max(1, hz)

    print(f"[main] running: {cfg_path} @ {hz} Hz")
    print(f"[main] debug visualize={visualize}, actuate={send_actuation}")

    last = time.time()
    while True:
        # 抓屏
        img = grabber.grab()
        mask, roi = seg.infer(img)
        pose = pose_est.estimate(mask, roi)

        # 控制
        cmd = agent.step(pose)

        # 下发（仅前台 Forza）
        if send_actuation and is_foreground_forza():
            act.send(cmd.steer, cmd.throttle, cmd.brake)

        # 可视化
        key = -1
        if visualize:
            overlay = img.copy()
            x, y, w, h = roi
            # 画 ROI
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 2)
            # 画掩膜
            m_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay[y:y+h, x:x+w] = cv2.addWeighted(overlay[y:y+h, x:x+w], 0.6, m_bgr, 0.4, 0)

            h_, w_ = overlay.shape[:2]
            txt1 = f"steer={cmd.steer:+.2f} thr={cmd.throttle:.2f} brk={cmd.brake:.2f}"
            txt2 = f"ey={pose.ey:+.2f}m epsi={pose.epsi:+.2f}rad v~{pose.v:.1f} m/s"
            txt3 = f"actuate={'ON' if send_actuation else 'OFF'}  (p to toggle)"
            cv2.putText(overlay, txt1, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(overlay, txt2, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(overlay, txt3, (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
            cv2.imshow("autodrive debug", overlay)
            key = cv2.waitKey(1) & 0xFF
        else:
            # 仍然读取键盘（OpenCV 窗口关闭时，无法读键。若需要，可改成 msvcrt）
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            send_actuation = not send_actuation
            print("[debug] actuate =", send_actuation)
        if key == ord(cfg["debug"]["save_frame_key"]):
            ts = int(time.time())
            cv2.imwrite(f"grab_{ts}.png", img)
            cv2.imwrite(f"mask_{ts}.png", mask)
            print("[debug] saved grab/mask at", ts)

        # 节流
        now = time.time()
        sleep_t = dt - (now - last)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last = now

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "autodrive/configs/base_stanley.yaml"
    main(cfg)
