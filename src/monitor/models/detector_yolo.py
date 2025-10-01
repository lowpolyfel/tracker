# detector yolo asincrono para cpu
# clases reales del modelo: bonder_tip, gold_reel
# mapeo logico interno: tip <- bonder_tip, reel <- gold_reel

import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np
from ultralytics import YOLO


class yolo_detector:
    # nombres reales del modelo y su mapeo a objetivos logicos
    _target_names = {
        "tip": {"bonder_tip"},
        "reel": {"gold_reel"},
    }

    def __init__(self, model_path: str, conf_thr: float = 0.25, imgsz: int = 416, debug: bool = True):
        # rutas y parametros
        self.model_path = model_path
        self.conf_thr = conf_thr
        self.imgsz = imgsz
        self.debug = debug

        # cargar modelo
        self.model = YOLO(self.model_path)

        # mapa id->nombre del modelo
        self.class_map = self._load_class_map()
        self.id_to_name = {int(k): str(v).lower() for k, v in self.class_map.items()}

        # resolver ids de clases objetivo a partir de nombres reales
        self.target_ids = self._resolve_target_ids()

        # colas e hilo
        self._in_q: Queue = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

        # estado de detecciones
        self._state = {"reel": None, "tip": None}

        if self.debug:
            print("[detector] clases del modelo:", self.id_to_name)
            print("[detector] ids objetivo:", self.target_ids)

    def _load_class_map(self):
        # intenta leer nombres de clases desde distintos lugares
        names = {}
        try:
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                names = {int(k): str(v) for k, v in self.model.names.items()}
        except Exception:
            pass
        if not names:
            try:
                inner = getattr(self.model, "model", None)
                if inner is not None and hasattr(inner, "names"):
                    d = inner.names
                    names = {int(k): str(v) for k, v in d.items()}
            except Exception:
                pass
        return names

    def _resolve_target_ids(self):
        resolved = {"reel": set(), "tip": set()}
        for logic, real_names in self._target_names.items():
            for cid, cname in self.id_to_name.items():
                if cname.lower() in real_names:
                    resolved[logic].add(int(cid))
        return resolved

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)

    def submit(self, frame_bgr: np.ndarray):
        # no bloquear si ya hay un frame pendiente
        if self._in_q.full():
            return
        h, w = frame_bgr.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        x0 = (self.imgsz - nw) // 2
        y0 = (self.imgsz - nh) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        meta = {"x0": x0, "y0": y0, "scale": scale, "orig_w": w, "orig_h": h}
        try:
            self._in_q.put((rgb, meta), block=False)
        except Exception:
            pass

    def get_state(self):
        # copia ligera del estado
        return {
            "reel": self._state["reel"].copy() if self._state["reel"] else None,
            "tip": self._state["tip"].copy() if self._state["tip"] else None,
        }

    def _worker(self):
        # hilo de inferencia no bloqueante con prints de depuracion
        while not self._stop.is_set():
            try:
                rgb, meta = self._in_q.get(timeout=0.1)
            except Empty:
                continue

            ts = time.time()
            try:
                results = self.model.predict(
                    source=rgb,
                    imgsz=self.imgsz,
                    conf=self.conf_thr,
                    verbose=False,
                    device="cpu",
                )
                if not results:
                    if self.debug:
                        print("[detector] no results")
                    self._update_none()
                    continue

                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is None or boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
                    if self.debug:
                        print("[detector] no boxes")
                    self._update_none()
                    continue

                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                conf = boxes.conf.detach().cpu().numpy().astype(float)

                if self.debug:
                    print("[detector] detecciones crudas:")
                    for i in range(len(xyxy)):
                        cname = self.id_to_name.get(int(cls[i]), str(int(cls[i])))
                        print(f"  id={int(cls[i])} name={cname} conf={conf[i]:.3f} bbox={xyxy[i].tolist()}")

                new_state = {"reel": None, "tip": None}

                for i in range(len(xyxy)):
                    cid = int(cls[i])
                    cname = self.id_to_name.get(cid, "")
                    label = None
                    if cid in self.target_ids.get("reel", set()):
                        label = "reel"
                    elif cid in self.target_ids.get("tip", set()):
                        label = "tip"
                    else:
                        if self.debug:
                            print(f"[detector] clase ignorada: {cname}")
                        continue

                    x1, y1, x2, y2 = xyxy[i]

                    x1 = (x1 - meta["x0"]) / meta["scale"]
                    y1 = (y1 - meta["y0"]) / meta["scale"]
                    x2 = (x2 - meta["x0"]) / meta["scale"]
                    y2 = (y2 - meta["y0"]) / meta["scale"]

                    x1 = max(0, min(meta["orig_w"] - 1, x1))
                    y1 = max(0, min(meta["orig_h"] - 1, y1))
                    x2 = max(0, min(meta["orig_w"] - 1, x2))
                    y2 = max(0, min(meta["orig_h"] - 1, y2))

                    item = {"bbox": (int(x1), int(y1), int(x2), int(y2)), "conf": float(conf[i]), "ts": ts}
                    prev = new_state.get(label)
                    if prev is None or item["conf"] > prev["conf"]:
                        new_state[label] = item
                        if self.debug:
                            print(f"[detector] {cname} -> {label} conf={item['conf']:.3f} bbox={item['bbox']}")

                self._state["reel"] = new_state["reel"]
                self._state["tip"] = new_state["tip"]

                if self.debug:
                    print(f"[detector] estado final: {self._state}")

            except Exception as e:
                if self.debug:
                    print("[detector] error en prediccion:", e)
                self._update_none()

    def _update_none(self):
        self._state["reel"] = None
        self._state["tip"] = None

