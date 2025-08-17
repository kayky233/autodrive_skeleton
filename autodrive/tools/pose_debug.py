# autodrive/tools/pose_debug.py
import math
from typing import Any, Dict, Optional, Iterable, Tuple, List

# ---- 运行期注入的显示窗口/ROI与最近一帧的轨迹像素 ----
_MONITOR = {"left": 0, "top": 0, "width": 0, "height": 0}
_ROI = {"y0": 0, "y1": 0}
_LAST_POLY_PX: List[Tuple[float, float]] = []   # [(x,y),...]
_LAST_REF_PX: Optional[Tuple[float, float]] = None
_LAST_REF_TAN: Optional[Tuple[float, float]] = None  # (dx,dy) 像素切向量

def set_viewport(monitor: Dict[str, int], roi: Dict[str, int]) -> None:
    """在 main 里把 env.monitor / env.roi 注入进来。"""
    _MONITOR.update(monitor or {})
    _ROI.update(roi or {})

def update_polyline_px(points: Iterable[Tuple[float, float]]) -> None:
    """在你画中线/轨迹的地方调用：update_polyline_px([(x1,y1), ...])"""
    global _LAST_POLY_PX, _LAST_REF_PX, _LAST_REF_TAN
    _LAST_POLY_PX = [(float(x), float(y)) for x, y in points] if points else []
    _LAST_REF_PX, _LAST_REF_TAN = None, None  # 让下一次提取时自动重算

def update_ref_px(pt: Tuple[float, float], tan: Optional[Tuple[float, float]] = None) -> None:
    """如果你手里已有参考点像素和切线，直接塞这里。"""
    global _LAST_REF_PX, _LAST_REF_TAN
    _LAST_REF_PX = (float(pt[0]), float(pt[1])) if pt else None
    _LAST_REF_TAN = (float(tan[0]), float(tan[1])) if tan else None

# ---------------- 工具 ----------------
def _norm_angle(a: float) -> float:
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def _get_opt(pose: Any, *names) -> Optional[float]:
    for n in names:
        if hasattr(pose, n):
            v = getattr(pose, n)
            return None if v is None else float(v)
        if isinstance(pose, dict) and n in pose:
            v = pose[n]
            return None if v is None else float(v)
    return None

def _default_car_px() -> Tuple[float, float]:
    """用 ROI 底边中心点作为车像素坐标的近似。"""
    W = _MONITOR.get("width", 0)
    y1 = _ROI.get("y1", 0)
    return float(W) * 0.5, float(y1)

def _tangent_of_polyline(points: List[Tuple[float, float]], idx: int) -> Tuple[float, float]:
    """取多段折线在 idx 附近的切向量（像素坐标系，右向为 +x，下向为 +y）。"""
    if not points:
        return (1.0, 0.0)
    if idx <= 0:
        x0, y0 = points[0]; x1, y1 = points[min(1, len(points)-1)]
    elif idx >= len(points) - 1:
        x0, y0 = points[max(0, len(points)-2)]; x1, y1 = points[-1]
    else:
        x0, y0 = points[idx]; x1, y1 = points[idx+1]
    return (x1 - x0, y1 - y0)

def _nearest_point_idx(points: List[Tuple[float, float]], x: float, y: float) -> int:
    best_i, best_d2 = 0, float("inf")
    for i, (px, py) in enumerate(points):
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i

# ---------------- 主入口 ----------------
def extract_pose_pixels(pose: Any) -> Dict[str, Optional[float]]:
    """
    优先从 pose 字段获取：
      px_car_x/px_car_y, px_ref_x/px_ref_y, yaw/ref_yaw 或 ref_dx/ref_dy
    若缺失，则：
      - 车像素 = ROI 底边中心
      - ref 像素 = 最近一次 update_ref_px()，或从 update_polyline_px() 给的折线里找车像素最近点
      - ref_yaw 像素 = 由折线切向量估算
    """
    # 1) car 像素
    cx = _get_opt(pose, "px_car_x", "cx", "x", "px_x")
    cy = _get_opt(pose, "px_car_y", "cy", "y", "px_y")
    if cx is None or cy is None:
        cx, cy = _default_car_px()

    # 2) ref 像素
    rx = _get_opt(pose, "px_ref_x", "rx")
    ry = _get_opt(pose, "px_ref_y", "ry")
    ref_yaw = _get_opt(pose, "ref_yaw", "ref_heading", "ref_theta")
    if (rx is None or ry is None) and _LAST_REF_PX is not None:
        rx, ry = _LAST_REF_PX
    if ref_yaw is None and _LAST_REF_TAN is not None:
        dx, dy = _LAST_REF_TAN
        ref_yaw = math.atan2(dy, dx)

    # 若还没有，尝试从最近的折线推一个
    if (rx is None or ry is None or ref_yaw is None) and _LAST_POLY_PX:
        idx = _nearest_point_idx(_LAST_POLY_PX, cx, cy)
        if rx is None or ry is None:
            rx, ry = _LAST_POLY_PX[idx]
        if ref_yaw is None:
            dx, dy = _tangent_of_polyline(_LAST_POLY_PX, idx)
            ref_yaw = math.atan2(dy, dx)

    # 3) 车头角（如果 pose 有世界系航向，就直接拿来对比）
    yaw = _get_opt(pose, "yaw", "heading", "theta", "hdg")

    yaw_norm = _norm_angle(yaw) if isinstance(yaw, (int, float)) else None
    ref_yaw_norm = _norm_angle(ref_yaw) if isinstance(ref_yaw, (int, float)) else None
    epsi_pix = _norm_angle(yaw - ref_yaw) if (isinstance(yaw, (int, float)) and isinstance(ref_yaw, (int, float))) else None

    return {
        "px_car_x": float(cx) if cx is not None else None,
        "px_car_y": float(cy) if cy is not None else None,
        "px_ref_x": float(rx) if rx is not None else None,
        "px_ref_y": float(ry) if ry is not None else None,
        "yaw": yaw,
        "ref_yaw": ref_yaw,
        "yaw_norm": yaw_norm,
        "ref_yaw_norm": ref_yaw_norm,
        "epsi_from_pixels": epsi_pix,
    }
