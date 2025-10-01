# fabrica de trackers opencv y gestor por clase
# tip: csrt + flujo optico lk
# reel: kcf o mosse
# kalman para suavizar y predecir cortos lapsos

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from .kalman import bbox_kalman


def _create_tracker_csrt():
    # csrt suele estar en cv2.legacy
    tr = None
    try:
        tr = cv2.legacy.TrackerCSRT_create()
    except Exception:
        try:
            tr = cv2.TrackerCSRT_create()
        except Exception:
            pass
    return tr


def _create_tracker_kcf():
    tr = None
    try:
        tr = cv2.legacy.TrackerKCF_create()
    except Exception:
        try:
            tr = cv2.TrackerKCF_create()
        except Exception:
            pass
    return tr


def _create_tracker_mosse():
    tr = None
    try:
        tr = cv2.legacy.TrackerMOSSE_create()
    except Exception:
        try:
            tr = cv2.TrackerMOSSE_create()
        except Exception:
            pass
    return tr


def _rect_from_bbox(b):
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def _bbox_from_rect(r):
    x, y, w, h = map(int, r)
    return (x, y, x + w, y + h)


class flow_refiner:
    # refinamiento con lk dentro de la bbox para tip
    def __init__(self, max_corners=40):
        self.prev_gray = None
        self.prev_pts = None
        self.max_corners = max_corners

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            self.reset()
            return bbox

        gray = cv2.cvtColor(frame, 6)  # 6 = cv2.COLOR_BGR2GRAY

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 6:
            roi = gray[y1:y2, x1:x2]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=4)
            if pts is not None:
                pts[:, 0, 0] += x1
                pts[:, 0, 1] += y1
            self.prev_gray = gray
            self.prev_pts = pts
            return bbox

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, winSize=(15, 15), maxLevel=2)
        good_new = next_pts[st == 1] if next_pts is not None else None
        good_old = self.prev_pts[st == 1] if self.prev_pts is not None else None

        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2) if good_new is not None and len(good_new) > 0 else None

        if good_new is None or len(good_new) < 4:
            return bbox

        xs = good_new[:, 0]
        ys = good_new[:, 1]
        nx1, ny1, nx2, ny2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
        pad = 2
        nx1, ny1 = max(0, nx1 - pad), max(0, ny1 - pad)
        nx2, ny2 = min(frame.shape[1] - 1, nx2 + pad), min(frame.shape[0] - 1, ny2 + pad)
        if nx2 <= nx1 or ny2 <= ny1:
            return bbox
        return (nx1, ny1, nx2, ny2)


class target_tracker:
    # gestor de seguimiento por objetivo
    def __init__(self, kind: str):
        self.kind = kind  # "tip" o "reel"
        self.tracker = None
        self.kalman = bbox_kalman(dt=1/30.0)
        self.flow = flow_refiner(max_corners=40) if kind == "tip" else None
        self.bbox = None
        self.ok_frames = 0
        self.missed = 0

    def init(self, frame, bbox):
        self.kalman.init_from_bbox(*bbox)
        self.bbox = bbox
        self.missed = 0
        self.ok_frames = 0
        self.flow.reset() if self.flow else None
        if self.kind == "tip":
            self.tracker = _create_tracker_csrt()
        else:
            self.tracker = _create_tracker_kcf() or _create_tracker_mosse()
        if self.tracker is not None:
            self.tracker.init(frame, _rect_from_bbox(bbox))

    def update_with_det(self, frame, bbox):
        self.init(frame, bbox)

    def update(self, frame):
        pred = self.kalman.predict()
        if self.tracker is None:
            if pred is not None:
                self.bbox = pred
            self.missed += 1
            return self.bbox, False

        ok, rect = self.tracker.update(frame)
        if not ok:
            self.missed += 1
            if pred is not None:
                self.bbox = pred
            return self.bbox, False

        bbox = _bbox_from_rect(rect)

        if self.flow is not None:
            bbox = self.flow.update(frame, bbox)

        smoothed = self.kalman.update(*bbox)
        self.bbox = smoothed if smoothed is not None else bbox
        self.ok_frames += 1
        self.missed = 0
        return self.bbox, True


class multi_target_tracking:
    # administra trackers para tip y reel y fusion con detecciones
    def __init__(self):
        self.trackers: Dict[str, target_tracker] = {
            "tip": target_tracker("tip"),
            "reel": target_tracker("reel"),
        }

    def update_from_detections(self, frame, state: dict):
        for k in ("tip", "reel"):
            item = state.get(k)
            if item is None:
                continue
            self.trackers[k].update_with_det(frame, item["bbox"])

    def step(self, frame):
        out = {}
        for k, t in self.trackers.items():
            bbox, ok = t.update(frame)
            out[k] = {"bbox": bbox, "ok": ok, "missed": t.missed}
        return out

