# autodrive/app/vis.py
import cv2
import numpy as np
from ..core.contracts import RoadMask, ControlCmd

def overlay_mask(frame_bgr, road: RoadMask):
    x, y, w, h = road.roi
    vis = frame_bgr.copy()
    mask3 = cv2.cvtColor(road.mask, cv2.COLOR_GRAY2BGR)
    vis[y:y+h, x:x+w] = cv2.addWeighted(vis[y:y+h, x:x+w], 0.7, mask3, 0.3, 0)
    return vis

def draw_cmd(img, cmd: ControlCmd, fps: float):
    h, w = img.shape[:2]
    # 方向条
    bar_w = int((cmd.steer + 1) * 0.5 * (w * 0.6))
    cv2.rectangle(img, (20, h-60), (20+bar_w, h-40), (0,255,0), -1)
    text = f"steer={cmd.steer:+.2f} thr={cmd.throttle:.2f} brk={cmd.brake:.2f} fps={fps:.1f}"
    cv2.putText(img, text, (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return img
