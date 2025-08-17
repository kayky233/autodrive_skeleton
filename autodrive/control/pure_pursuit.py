# autodrive/control/pure_pursuit.py
import math
from typing import Protocol

class Pose(Protocol):
    ey: float
    epsi: float
    v: float

class PurePursuit:
    """Minimal Pure Pursuit controller."""
    def __init__(self, L: float = 2.7, k_look: float = 0.6,
                 Ld_min: float = 6.0, steer_limit: float = 0.6,
                 epsi_gain: float = 0.2):
        self.L = float(L)
        self.k_look = float(k_look)
        self.Ld_min = float(Ld_min)
        self.steer_limit = float(steer_limit)
        self.epsi_gain = float(epsi_gain)

    def steer(self, pose: Pose) -> float:
        v = max(0.0, float(getattr(pose, "v", 0.0)))
        ey = float(getattr(pose, "ey", 0.0))
        epsi = float(getattr(pose, "epsi", 0.0))

        Ld = max(self.Ld_min, self.k_look * v)
        kappa = 2.0 * ey / max(1e-3, Ld * Ld) + self.epsi_gain * epsi
        delta = math.atan(self.L * kappa)

        if delta > self.steer_limit:
            delta = self.steer_limit
        elif delta < -self.steer_limit:
            delta = -self.steer_limit
        return float(delta)
