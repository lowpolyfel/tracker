# integra yolo para deteccion y trackers para seguimiento continuo
# la vista de camara se mantiene fluida; yolo y tracking corren en paralelo

import sys
import time

from monitor.ui.startup_screen import (
    show_waiting_for_camera,
    show_error_retry,
    show_camera_preview,
    draw_detection_overlay,
    draw_tracking_overlay,
)
from monitor.devices.camera import find_camera_index, open_camera
from monitor.models.detector_yolo import yolo_detector
from monitor.tracking.trackers import multi_target_tracking


def main():
    model_path = "y8n_640_ep80/weights/best.pt"

    detector = yolo_detector(model_path=model_path, conf_thr=0.25, imgsz=416, debug=False)
    mtt = multi_target_tracking()

    while 1:
        show_waiting_for_camera()

        cam_index = find_camera_index(max_index=5)
        if cam_index < 0:
            action = show_error_retry(
                error_text="no se detecto camara usb",
                instructions="pulsa r para reintentar o q para salir",
            )
            if action == "retry":
                continue
            sys.exit(1)

        cap = open_camera(cam_index)
        if cap is None:
            action = show_error_retry(
                error_text=f"no se pudo abrir la camara en el indice {cam_index}",
                instructions="pulsa r para probar de nuevo o q para salir",
            )
            if action == "retry":
                continue
            sys.exit(2)

        detector.start()

        def overlay_fn(frame):
            t = time.time()
            if int(t * 30) % 3 == 0:
                detector.submit(frame)

            det_state = detector.get_state()
            if det_state.get("tip") or det_state.get("reel"):
                mtt.update_from_detections(frame, det_state)

            trk_state = mtt.step(frame)

            draw_detection_overlay(frame, det_state, conf_min=0.25)
            draw_tracking_overlay(frame, trk_state)

        next_action = show_camera_preview(
            cap,
            overlay_text="detector y tracking activos... (q para salir)",
            overlay_fn=overlay_fn,
        )

        detector.stop()

        if next_action == "back":
            continue
        break


if __name__ == "__main__":
    main()

