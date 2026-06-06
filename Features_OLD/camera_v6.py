#v6
import numpy as np
import math


class Camera3D:
    def __init__(self, pos=(0.0, 240.0, 580.0), pitch=-22.0, yaw=0.0):
        self.position = np.array(pos, dtype=float)
        self.pitch    = float(pitch)   # degrees, clamped ±88
        self.yaw      = float(yaw)     # degrees, free
        self.fov      = 55.0           # vertical field of view in degrees

    def rotate(self, dyaw: float, dpitch: float):
        self.yaw  -= dyaw
        self.pitch = float(np.clip(self.pitch - dpitch, -88.0, 88.0))

    def move(self, right_amt: float, up_amt: float, forward_amt: float):
        world_up = np.array([0.0, 1.0, 0.0])
        forward, right, _ = self._basis()
        self.position += right * right_amt + world_up * up_amt + forward * forward_amt

    def _basis(self):
        pr = math.radians(self.pitch)
        yr = math.radians(self.yaw)
        forward = np.array([
            math.cos(pr) * math.sin(yr),
            -math.sin(pr),
            -math.cos(pr) * math.cos(yr)
        ], dtype=float)

        world_up = np.array([0.0, 1.0, 0.0])
        right    = np.cross(forward, world_up)
        norm_r   = np.linalg.norm(right)
        if norm_r < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= norm_r

        up_cam = np.cross(right, forward)
        return forward, right, up_cam

    def _proj_params(self, W: int, H: int):
        f   = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        asp = W / H
        return f, asp
        
    def get_screen_radius(self, radius_world, depth, W, H):
        if depth <= 0: return 0
        f, asp = self._proj_params(W, H)
        return radius_world * f / depth * (H / 2)

    def project_single(self, pt, W: int, H: int):
        forward, right, up_cam = self._basis()
        p   = np.asarray(pt, float) - self.position
        px  = float(p @ right)
        py  = float(p @ up_cam)
        pz  = float(p @ forward)
        if pz <= 0.1:
            return None

        f, asp = self._proj_params(W, H)
        sx = ( px * f / (pz * asp)) * W / 2 + W / 2
        sy = (-py * f /  pz        ) * H / 2 + H / 2
        return sx, sy, pz

    def project_batch(self, pts: np.ndarray, W: int, H: int):
        forward, right, up_cam = self._basis()
        p   = pts - self.position           # (N, 3)
        px  = p @ right
        py  = p @ up_cam
        pz  = p @ forward

        mask = pz > 0.1
        f, asp = self._proj_params(W, H)

        sx = np.zeros(len(pts))
        sy = np.zeros(len(pts))
        if mask.any():
            sx[mask] = ( px[mask] * f / (pz[mask] * asp)) * W / 2 + W / 2
            sy[mask] = (-py[mask] * f /  pz[mask]        ) * H / 2 + H / 2

        return sx, sy, pz, mask
