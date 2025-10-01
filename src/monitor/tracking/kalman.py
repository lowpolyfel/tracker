# filtro kalman simple para bbox con modelo de velocidad constante
# estado: [cx, cy, w, h, vx, vy, vw, vh]

import numpy as np


class bbox_kalman:
    def __init__(self, dt: float = 1.0, process_var: float = 1e-2, meas_var: float = 1e-1):
        self.dt = dt

        self.x = np.zeros((8, 1), dtype=np.float32)

        self.f = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.f[i, i + 4] = dt

        self.h = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.h[i, i] = 1.0

        q_base = process_var
        self.q = np.eye(8, dtype=np.float32) * q_base
        for i in range(4, 8):
            self.q[i, i] = q_base * 10.0

        self.r = np.eye(4, dtype=np.float32) * meas_var

        self.p = np.eye(8, dtype=np.float32)

        self.inited = False

    def init_from_bbox(self, x1: int, y1: int, x2: int, y2: int):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.x[:] = 0.0
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.x[2, 0] = w
        self.x[3, 0] = h
        self.p = np.eye(8, dtype=np.float32)
        self.inited = True

    def predict(self):
        if not self.inited:
            return None
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.get_bbox()

    def update(self, x1: int, y1: int, x2: int, y2: int):
        if not self.inited:
            self.init_from_bbox(x1, y1, x2, y2)
            return self.get_bbox()
        z = np.zeros((4, 1), dtype=np.float32)
        z[0, 0] = (x1 + x2) * 0.5
        z[1, 0] = (y1 + y2) * 0.5
        z[2, 0] = max(1.0, float(x2 - x1))
        z[3, 0] = max(1.0, float(y2 - y1))

        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        y = z - (self.h @ self.x)
        self.x = self.x + k @ y
        i = np.eye(8, dtype=np.float32)
        self.p = (i - k @ self.h) @ self.p
        return self.get_bbox()

    def get_bbox(self):
        if not self.inited:
            return None
        cx, cy, w, h = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])
        x1 = int(cx - w * 0.5)
        y1 = int(cy - h * 0.5)
        x2 = int(cx + w * 0.5)
        y2 = int(cy + h * 0.5)
        return x1, y1, x2, y2

