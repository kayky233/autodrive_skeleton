# autodrive/agents/base_agent.py
# ── (A) 顶部导入处，把原来的 import 替换为下面三行 ──
from dataclasses import dataclass
from typing import Optional, Dict, Any
import csv, time, os

# 用于把上游像素轨迹/位置打印与可视化
try:
    from autodrive.tools.pose_debug import update_polyline_px, print_pose_line
except Exception:
    def update_polyline_px(_):  # 兜底，不影响运行
        pass
    def print_pose_line(_):     # 兜底，不影响运行
        pass


@dataclass
class ControlCmd:
    steer: float
    throttle: float
    brake: float = 0.0


@dataclass
class SpeedCtrlCfg:
    vmax: float = 12.0
    kp: float = 0.12
    ki: float = 0.0
    kd: float = 0.0
    throttle_limit: float = 0.7
    brake_limit: float = 0.9


class BaseAgent:
    def __init__(self, controller_type: str = "stanley",
                 ctl_params: Optional[Dict[str, Any]] = None,
                 speed_cfg: Optional[SpeedCtrlCfg] = None):
        ctl_params = ctl_params or {}
        self.controller_type = controller_type.lower().strip()

        if self.controller_type == "stanley":
            from autodrive.control.stanley import Stanley
            self.lat = Stanley(**ctl_params)
        elif self.controller_type in ("pp", "pure_pursuit", "pure-pursuit"):
            from autodrive.control.pure_pursuit import PurePursuit
            self.lat = PurePursuit(**ctl_params)
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        self.spd = speed_cfg or SpeedCtrlCfg()
        self._ei = 0.0
        self._prev_err = 0.0

        # 诊断 CSV 懒初始化
        self._diag_ready = False
        self._diag_path = "control_total_diag.csv"
        self._diag_fh = None
        self._diag_writer = None

    def _speed_control(self, v: float):
        err = max(0.0, self.spd.vmax) - max(0.0, v)
        self._ei += err
        de = err - self._prev_err
        self._prev_err = err

        u = self.spd.kp*err + self.spd.ki*self._ei + self.spd.kd*de
        if u >= 0.0:
            throttle = min(self.spd.throttle_limit, u)
            return throttle, 0.0
        else:
            brake = min(self.spd.brake_limit, -u)
            return 0.0, brake

    def step(self, pose):
        """pose 需至少包含: ey / epsi / v；可选: kappa / kappa_ahead / poly_px"""
        delta = self.lat.steer(pose)  # 控制器输出（弧度）
        throttle, brake = self._speed_control(getattr(pose, "v", 0.0))

        # 维测：统一用 delta 作为“控制器内部输出/最终转向命令”
        delta_ctl = delta   # 控制器限幅后的内部输出
        steer_cmd = delta   # 外层映射到平台（此处等同）

        # 写一行诊断 + 把 polyline（若有）转发给调试层绘制
        self._log_diag(
            pose,
            delta_ctl=delta_ctl,
            steer_cmd=steer_cmd,
            throttle=throttle,
            brake=brake
        )
        return ControlCmd(steer=delta, throttle=throttle, brake=brake)

    # ----------------- 下面是诊断记录 -----------------
    def _init_diag(self, path: str = "control_total_diag.csv"):
        """懒初始化：第一次调用 _log_diag 时才建表头。"""
        if getattr(self, "_diag_ready", False):
            return
        self._diag_path = path
        self._diag_fh = None
        self._diag_writer = None

        first_open = False
        try:
            try:
                with open(self._diag_path, "r", newline="", encoding="utf-8") as _:
                    pass
            except FileNotFoundError:
                first_open = True

            self._diag_fh = open(self._diag_path, "a", newline="", encoding="utf-8")
            self._diag_writer = csv.writer(self._diag_fh)
            if first_open:
                self._diag_writer.writerow([
                    # 时间戳 & Pose 输入
                    "ts_ms","ey","epsi","v","kappa","kappa_ahead",
                    # 控制器内部（如可用）
                    "cross","hdg","ff","delta_raw","delta_ctl",
                    # 外层映射后的输出
                    "steer_cmd","throttle","brake"
                ])
                self._diag_fh.flush()
            self._diag_ready = True
        except Exception:
            self._diag_ready = False
            self._diag_writer = None
            self._diag_fh = None

    def _log_diag(self, pose, delta_ctl: float, steer_cmd: float, throttle: float, brake: float):
        """把 Pose + 控制器中间量 + 最终输出打一行，同时把像素轨迹线转发给调试层。"""
        if not getattr(self, "_diag_ready", False):
            self._init_diag()

            # ── (B) 在 BaseAgent._log_diag() 的开头，update_polyline_px(poly_px) 之后加上这一小段 ──
            # 可视化：若 pose 携带轨迹像素折线，则转发给调试层画出来
            try:
                poly_px = getattr(pose, "poly_px", None)
                if poly_px:
                    update_polyline_px(poly_px)
            except Exception:
                pass

            # 每帧位置维测打印（export AD_PRINT_POSE=0 可关闭）
            try:
                if os.environ.get("AD_PRINT_POSE", "1") == "1":
                    print_pose_line(pose)
            except Exception:
                pass

        if self._diag_writer is None:
            return

        ts_ms = int(time.time() * 1000.0)
        ey   = float(getattr(pose, "ey", 0.0))
        epsi = float(getattr(pose, "epsi", 0.0))
        v    = float(getattr(pose, "v", 0.0))
        # kappa / kappa_ahead 缺失时记 0.0，避免 None 转 float 的异常
        kappa = float(getattr(pose, "kappa", 0.0) or 0.0)
        kappa_ahead = float(getattr(pose, "kappa_ahead", 0.0) or 0.0)

        # 控制器内部诊断：若可用则取值，不可用留空
        cross = hdg = ff = delta_raw = ""
        try:
            if hasattr(self.lat, "last_diag") and isinstance(self.lat.last_diag, dict):
                d = self.lat.last_diag
                cross = d.get("cross", "")
                hdg = d.get("hdg", "")
                ff = d.get("ff", "")
                delta_raw = d.get("delta_raw", "")
        except Exception:
            pass

        try:
            self._diag_writer.writerow([
                ts_ms, ey, epsi, v, kappa, kappa_ahead,
                cross, hdg, ff, delta_raw, delta_ctl,
                steer_cmd, throttle, brake
            ])
            self._diag_fh.flush()
        except Exception:
            pass

    def __del__(self):
        try:
            if self._diag_fh:
                self._diag_fh.close()
        except Exception:
            pass
