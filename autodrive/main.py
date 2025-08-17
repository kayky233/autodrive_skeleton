# autodrive/main.py
from __future__ import annotations
import sys
import os
import signal
import yaml
from pathlib import Path

# === import project modules ===
from .app.loop import App
from .core.contracts import Frame
from .core.interfaces import IEnv, IPerception, IStateEstimator, IPlanner, IController, IActuator

# Envs
from .env.fh5_dxgi import FH5Env
from .env.fake_env import FakeEnv  # 可用于本地回放测试

# Perception
from .perception.lane_cv import LaneCV
from .perception.seg_onnx import SegONNX

# State
from .state.kinematics import SimpleKinematics

# Planning
from .planning.path_follow_pp import PurePursuit
from .planning.path_follow_stanley import Stanley

# Control
from .control.pid_speed import BlendController

# Actuators
from .actuators.gamepad_vigem import GamepadAct
from .actuators.keyboard_sendinput import KeyboardAct
import time, cv2
from .app.vis import overlay_mask, draw_cmd
from autodrive.tools.find_fh5_window import get_foreground_window_rect


def load_cfg(path: str | Path):
    p = Path(path)
    if not p.is_absolute():
        # 优先：调用处的工作目录
        if not p.exists():
            pkg_dir = Path(__file__).resolve().parent              # .../autodrive
            cand = pkg_dir / p                                     # 包内相对路径
            if cand.exists():
                p = cand
            else:
                proj_root = pkg_dir.parent                         # 项目根
                cand2 = proj_root / p
                if cand2.exists():
                    p = cand2
                else:
                    raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg) -> IEnv:
    env_cfg = cfg["env"]
    typ = env_cfg.get("type", "fh5_dxgi").lower()
    if typ == "fh5_dxgi":
        mon = env_cfg.get("monitor", {"left": 140, "top": 120, "width": 1600, "height": 900})
        return FH5Env(monitor=mon, bgr=True)
    elif typ == "fake":
        # 仅离线测试：在 configs/base.yaml 里指定 frames_glob
        pattern = env_cfg.get("frames_glob", "./frames/*.jpg")
        fps = float(env_cfg.get("fps", 30))
        return FakeEnv(pattern=pattern, fps=fps, loop=True)
    else:
        raise ValueError(f"Unknown env type: {typ}")


def make_perception(cfg) -> IPerception:
    pc = cfg["perception"]
    typ = pc.get("type", "lane_cv").lower()
    env_cfg = cfg.get("env", {})
    roi_cfg = env_cfg.get("roi", {}) if isinstance(env_cfg.get("roi", {}), dict) else {}
    y0 = int(roi_cfg.get("y0", 500))   # 默认 500
    y1 = int(roi_cfg.get("y1", 880))   # 默认 880

    if typ == "lane_cv":
        lower = tuple(pc.get("hsv_lower", (0, 0, 80)))
        upper = tuple(pc.get("hsv_upper", (180, 60, 255)))
        return LaneCV(roi_y0=y0, roi_y1=y1, hsv_lower=lower, hsv_upper=upper)

    if typ == "seg_onnx":
        return SegONNX(
            model_path=pc.get("model_path"),
            input_size=tuple(pc.get("input_size", (512, 288))),
            conf=float(pc.get("conf", 0.5)),
            roi_y0=y0,
            roi_y1=y1,
        )
    raise ValueError(f"Unknown perception type: {typ}")



def make_state(cfg) -> IStateEstimator:
    st = cfg["state"]
    return SimpleKinematics(
        pix2meter=float(st.get("pix2meter", 0.02)),
        heading_gain=float(st.get("heading_gain", 1.0)),
    )


def make_planner(cfg) -> IPlanner:
    pl = cfg["planner"]
    typ = pl.get("type", "pure_pursuit").lower()
    if typ == "pure_pursuit":
        return PurePursuit(
            L=float(pl.get("L", 2.6)),
            k1=float(pl.get("k1", 2.0)),
            k2=float(pl.get("k2", 0.3)),
            vmax=float(pl.get("vmax", 25.0)),
            curv_gain=float(pl.get("curv_gain", 30.0)),
        )
    if typ == "stanley":
        return Stanley(
            k=float(pl.get("k", 1.0)),
            vmax=float(pl.get("vmax", 25.0)),
            curv_gain=float(pl.get("curv_gain", 30.0)),
        )
    raise ValueError(f"Unknown planner type: {typ}")


def make_controller(cfg) -> IController:
    ct = cfg["controller"]
    typ = ct.get("type", "blend").lower()
    if typ == "blend":
        return BlendController(
            max_steer_rate=float(ct.get("max_steer_rate", 1.5)),
            max_cmd_rate=float(ct.get("max_cmd_rate", 2.0)),
        )
    raise ValueError(f"Unknown controller type: {typ}")


def make_actuator(cfg) -> IActuator:
    ac = cfg["actuator"]
    typ = ac.get("type", "gamepad_vigem").lower()
    if typ == "gamepad_vigem":
        return GamepadAct()
    if typ == "keyboard":
        return KeyboardAct()
    raise ValueError(f"Unknown actuator type: {typ}")


def main(cfg_path: str = "configs/base.yaml"):
    cfg = load_cfg(cfg_path)
    env = make_env(cfg)
    perc = make_perception(cfg)
    st = make_state(cfg)
    plan = make_planner(cfg)
    ctrl = make_controller(cfg)
    act = make_actuator(cfg)

    print("[debug] env.monitor:", cfg["env"]["monitor"])
    print("[debug] env.roi:", cfg["env"]["roi"])

    loop_hz = int(cfg.get("loop", {}).get("hz", 40))
    log_path = cfg.get("logging", {}).get("csv", "logs/online.csv")

    app = App(env, perc, st, plan, ctrl, act, hz=loop_hz, log_path=log_path)

    # 优雅退出
    def _graceful_exit(*_):
        try:
            if app.logger:
                app.logger.close()
        finally:
            print("\n[main] stopped.")
            os._exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    print(f"[main] running: {cfg_path}  @ {loop_hz} Hz")
    print(f"[main] logging to: {log_path}")
    dbg = cfg.get("debug", {})
    viz = bool(dbg.get("visualize", True))
    send_actuation = bool(dbg.get("actuate", False))  # 先不控车，按 p 可切换
    hz = loop_hz
    period = 1.0 / hz
    last = time.perf_counter()
    fps = 0.0

    print(f"[main] debug visualize={viz}, actuate={send_actuation}")
    while True:
        frame = env.grab()
        road = perc.infer(frame)
        pose, curv = st.estimate(frame, road)
        desired = plan.plan(pose, curv)
        cmd = ctrl.regulate(desired, pose)

        if send_actuation:
            act.send(cmd)

        if app.logger:
            app.logger.log(frame.ts, pose, curv, cmd)

        now = time.perf_counter()
        dt = now - last
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        last = now

        if viz:
            vis = overlay_mask(frame.rgb, road)
            vis = draw_cmd(vis, cmd, fps)
            cv2.imshow("autodrive debug", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break  # 退出
            if key == ord('v'): viz = not viz  # 切换可视化
            if key == ord('p'):  # 运行中切换是否下发手柄
                rect = get_foreground_window_rect()
                print("[debug] current foreground window rect:", rect)
                send_actuation = not send_actuation

        time.sleep(max(0, period - (time.perf_counter() - now)))
    cv2.destroyAllWindows()
    if app.logger: app.logger.close()


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    main(cfg_file)
