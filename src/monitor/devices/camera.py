# deteccion y apertura de camaras usb con opencv
# evita constantes en mayusculas usando ids numericos

import cv2


def find_camera_index(max_index: int = 5) -> int:
    # recorre indices 0..max_index y valida lectura
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return i
    return -1


def open_camera(index: int):
    # abre camara y configura resolucion y fps basicos
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return 0

    # 3 = width, 4 = height, 5 = fps
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(5, 30)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return 0
    return cap
