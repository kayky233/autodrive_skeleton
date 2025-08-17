# autodrive/control/stanley.py
import math
import time
import csv
from typing import Protocol, Optional, Dict, Any, TextIO


class Pose(Protocol):
    # —— 控制所需的最小字段 ——
    ey: float                 # 横向误差 [m]
    epsi: float               # 航向误差 [rad]
    v: float                  # 速度 [m/s]
    # 前馈曲率（可选）
    kappa: Optional[float]
    kappa_ahead: Optional[float]

    # —— 下面这些都是可选，仅用于诊断/像素对齐，可没有 ——
    yaw: Optional[float]          # 车辆绝对航向（rad）
    ref_yaw: Optional[float]      # 参考切向航向（rad）
    epsi_pix: Optional[float]     # 像素系下的航向误差（可选）

    px_car_x: Optional[float]     # 车辆像素坐标（通常是车底中心）
    px_car_y: Optional[float]
    px_ref_x: Optional[float]     # 参考轨迹足点的像素坐标
    px_ref_y: Optional[float]


class Stanley:
    """
    经典 Stanley（带前馈曲率），算法不变，仅加入可选的：
      - 死区（deadband）
      - 三项符号修正（ey_sign / hdg_sign / ff_sign）
      - 输出映射（steer_sign / steer_bias）
      - 角度稳健处理（若提供 yaw/ref_yaw，则自动对齐参考方向并归一化）
      - 诊断增强（像素位姿/参考参数：px_car/px_ref/yaw/ref_yaw/epsi_pix）

    数学形式（在默认参数=1.0/0.0 时与原实现等价）：
      cross = atan( k * (ey_sign*ey) / (v + v_soft) )
      hdg   = epsi_gain * (hdg_sign*epsi_used)   # epsi_used 为归一化/纠偏后的航向误差
      ff    = ff_gain * atan( L * (ff_sign * kappa_ff) )
      delta_raw = cross + hdg + ff
      delta_db  = 0 (if |delta_raw| < deadband else delta_raw)
      delta_clip = clip(delta_db, [-steer_limit, +steer_limit])
      delta_out  = steer_sign * delta_clip + steer_bias
      delta      = clip(delta_out, [-steer_limit, +steer_limit])   # 最终输出
    """

    def __init__(
        self,
        L: float = 2.7,
        k: float = 2.0,
        epsi_gain: float = 1.2,
        v_soft: float = 0.3,
        ff_gain: float = 1.0,
        steer_limit: float = 0.8,
        # === 新增：行为保持默认不变 ===
        deadband: float = 0.0,      # 转角死区（弧度）
        ey_sign: float = 1.0,       # 横向误差符号修正
        hdg_sign: float = 1.0,      # 航向误差符号修正
        ff_sign: float = 1.0,       # 前馈曲率符号修正
        steer_sign: float = 1.0,    # 最终输出到平台的方向映射
        steer_bias: float = 0.0,    # 最终输出的常值偏置（弧度）
        # 维测相关（默认不写文件，只把最后一次诊断放到 self.last_diag）
        diag_csv_path: Optional[str] = None,
        diag_print_pose: bool = False,        # 每帧打印 [pose] 行
        # 可选软夹限（默认 None = 不启用，不影响原行为）
        ey_clip_max: Optional[float] = None,        # 例：5.0（m）
        epsi_clip_rad: Optional[float] = None       # 例：math.pi/2
    ):
        self.L = float(L)
        self.k = float(k)
        self.epsi_gain = float(epsi_gain)
        self.v_soft = float(max(0.01, v_soft))
        self.ff_gain = float(ff_gain)
        self.steer_limit = float(steer_limit)

        # 新增参数（默认不改动行为）
        self.deadband = float(max(0.0, deadband))
        self.ey_sign = float(ey_sign)
        self.hdg_sign = float(hdg_sign)
        self.ff_sign = float(ff_sign)
        self.steer_sign = float(steer_sign)
        self.steer_bias = float(steer_bias)

        # 诊断：最近一次的中间量
        self.last_diag: Dict[str, Any] = {}

        # 诊断：可选写 CSV
        self._diag_csv_path = diag_csv_path
        self._diag_csv_fp: Optional[TextIO] = None
        self._diag_csv_writer: Optional[csv.writer] = None

        # 可选打印/夹限
        self._diag_print_pose = bool(diag_print_pose)
        self._ey_clip_max = float(ey_clip_max) if ey_clip_max is not None else None
        self._epsi_clip_rad = float(epsi_clip_rad) if epsi_clip_rad is not None else None

        if self._diag_csv_path:
            self._open_diag_csv()

    # -------------------- 内部工具 --------------------
    @staticmethod
    def _normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _num_or_blank(x: Any):
        try:
            if x is None:
                return ""
            v = float(x)
            if math.isfinite(v):
                return v
            return ""
        except Exception:
            return ""

    def _open_diag_csv(self):
        # 追加写，首开时打表头
        first_open = False
        try:
            try:
                with open(self._diag_csv_path, "r", newline="", encoding="utf-8") as _:
                    pass
            except FileNotFoundError:
                first_open = True

            self._diag_csv_fp = open(self._diag_csv_path, "a", newline="", encoding="utf-8")
            self._diag_csv_writer = csv.writer(self._diag_csv_fp)
            if first_open:
                # ✨ 表头里加入像素/航向等可视诊断字段
                self._diag_csv_writer.writerow([
                    "ts_ms",
                    "ey", "epsi_norm", "v",
                    "kappa", "kappa_ahead",
                    "cross", "hdg", "ff",
                    "delta_raw", "delta_db", "delta_clip", "delta_mapped(final)",
                    # 可视诊断
                    "px_car_x", "px_car_y",
                    "px_ref_x", "px_ref_y",
                    "yaw", "ref_yaw", "epsi_pix"
                ])
                self._diag_csv_fp.flush()
        except Exception:
            self._diag_csv_fp = None
            self._diag_csv_writer = None

    def _diag_write_row(self, row):
        if self._diag_csv_writer is None:
            return
        try:
            self._diag_csv_writer.writerow(row)
            self._diag_csv_fp.flush()
        except Exception:
            pass

    def _compute_epsi_used(self, pose: Pose):
        """
        优先使用 yaw/ref_yaw 来稳健计算 epsi：
          1) epsi0 = wrap(yaw - ref_yaw)
          2) 若 |epsi0| > 90°，说明参考方向反了 → ref_yaw += π，再算一次
        否则回退到 pose.epsi 并做 wrap。
        返回: (epsi_used, yaw_val, ref_yaw_val, used_from_yaw)
        """
        yaw_val = getattr(pose, "yaw", None)
        ref_yaw_val = getattr(pose, "ref_yaw", None)
        used_from_yaw = False

        if isinstance(yaw_val, (int, float)) and isinstance(ref_yaw_val, (int, float)):
            if math.isfinite(yaw_val) and math.isfinite(ref_yaw_val):
                epsi0 = self._normalize_angle(float(yaw_val) - float(ref_yaw_val))
                if abs(epsi0) > (math.pi * 0.5):  # 参考方向反了，纠正
                    ref_yaw_val = self._normalize_angle(float(ref_yaw_val) + math.pi)
                    epsi0 = self._normalize_angle(float(yaw_val) - float(ref_yaw_val))
                used_from_yaw = True
                epsi_used = epsi0
            else:
                epsi_used = self._normalize_angle(float(getattr(pose, "epsi", 0.0)))
        else:
            epsi_used = self._normalize_angle(float(getattr(pose, "epsi", 0.0)))

        # 可选夹限（例如 π/2），避免极端值把控制打爆
        if self._epsi_clip_rad is not None:
            lim = abs(self._epsi_clip_rad)
            if epsi_used > lim:
                epsi_used = lim
            elif epsi_used < -lim:
                epsi_used = -lim

        return epsi_used, yaw_val, ref_yaw_val, used_from_yaw

    # -------------------- 控制主过程 --------------------
    def steer(self, pose: Pose) -> float:
        # 读取输入
        v = max(0.0, float(getattr(pose, "v", 0.0)))
        ey = float(getattr(pose, "ey", 0.0))

        # 可选对 ey 软夹限（不改变原默认行为）
        if self._ey_clip_max is not None:
            lim = abs(self._ey_clip_max)
            if ey > lim:
                ey = lim
            elif ey < -lim:
                ey = -lim

        # 航向误差：更稳健的“从 yaw/ref_yaw 计算”，否则用 pose.epsi
        epsi_used, yaw_val, ref_yaw_val, used_from_yaw = self._compute_epsi_used(pose)

        # 三项（带可选符号修正），默认=1.0时与原实现一致
        cross = math.atan2(self.k * (self.ey_sign * ey), v + self.v_soft)
        hdg   = self.epsi_gain * (self.hdg_sign * epsi_used)

        kappa_ff = float(getattr(pose, "kappa_ahead", getattr(pose, "kappa", 0.0)))
        ff = self.ff_gain * math.atan(self.L * (self.ff_sign * kappa_ff))

        # 原始和组合
        delta_raw = cross + hdg + ff

        # 死区
        if abs(delta_raw) < self.deadband:
            delta_db = 0.0
        else:
            delta_db = delta_raw

        # 限幅
        if delta_db > self.steer_limit:
            delta_clip = self.steer_limit
        elif delta_db < -self.steer_limit:
            delta_clip = -self.steer_limit
        else:
            delta_clip = delta_db

        # 输出映射（平台方向/安装/接口差异）
        delta_out = self.steer_sign * delta_clip + self.steer_bias

        # 最终限幅（避免 bias 后超限）
        if delta_out > self.steer_limit:
            delta = self.steer_limit
        elif delta_out < -self.steer_limit:
            delta = -self.steer_limit
        else:
            delta = delta_out

        # ===== 诊断收集 =====
        ts_ms = int(time.time() * 1000.0)
        kappa_val = float(getattr(pose, "kappa", 0.0))
        kappa_ahead_val = float(getattr(pose, "kappa_ahead", 0.0))

        # 像素/显示相关（可为空）
        px_car_x = getattr(pose, "px_car_x", None)
        px_car_y = getattr(pose, "px_car_y", None)
        px_ref_x = getattr(pose, "px_ref_x", None)
        px_ref_y = getattr(pose, "px_ref_y", None)
        epsi_pix = getattr(pose, "epsi_pix", None)

        self.last_diag = {
            "ts_ms": ts_ms,
            # 控制核心
            "ey": ey,
            "epsi_norm": epsi_used,   # 实际用于控制的航向误差（归一化/纠偏后）
            "v": v,
            "kappa": kappa_val,
            "kappa_ahead": kappa_ahead_val,
            "cross": cross,
            "hdg": hdg,
            "ff": ff,
            "delta_raw": delta_raw,
            "deadband": self.deadband,
            "delta_db": delta_db,
            "delta_clip": delta_clip,
            "steer_sign": self.steer_sign,
            "steer_bias": self.steer_bias,
            "delta": delta,           # 最终输出
            "limit": self.steer_limit,
            # 符号修正追踪
            "ey_sign": self.ey_sign,
            "hdg_sign": self.hdg_sign,
            "ff_sign": self.ff_sign,
            # 可视诊断
            "px_car_x": px_car_x,
            "px_car_y": px_car_y,
            "px_ref_x": px_ref_x,
            "px_ref_y": px_ref_y,
            "yaw": yaw_val,
            "ref_yaw": ref_yaw_val,
            "epsi_pix": epsi_pix,
            "epsi_from_yaw": used_from_yaw,
        }

        # 可选写 CSV（把用于控制的 epsi 写成 epsi_norm；同时写像素/航向等诊断）
        if self._diag_csv_writer:
            self._diag_write_row([
                ts_ms,
                ey, epsi_used, v,
                kappa_val, kappa_ahead_val,
                cross, hdg, ff,
                delta_raw, delta_db, delta_clip, delta,
                self._num_or_blank(px_car_x), self._num_or_blank(px_car_y),
                self._num_or_blank(px_ref_x), self._num_or_blank(px_ref_y),
                self._num_or_blank(yaw_val), self._num_or_blank(ref_yaw_val),
                self._num_or_blank(epsi_pix),
            ])

        # 可选：在控制侧也输出一行 [pose]，方便你对齐游戏画面
        if self._diag_print_pose:
            def f3(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return "NA"

            print(
                "[pose] "
                f"px_car=({f3(px_car_x)},{f3(px_car_y)}) "
                f"px_ref=({f3(px_ref_x)},{f3(px_ref_y)}) "
                f"yaw={f3(yaw_val)} ref_yaw={f3(ref_yaw_val)} "
                f"epsi_used={f3(epsi_used)} epsi_pix={f3(epsi_pix)} ey={f3(ey)}"
            )

        return float(delta)

    def __del__(self):
        try:
            if self._diag_csv_fp:
                self._diag_csv_fp.close()
        except Exception:
            pass
