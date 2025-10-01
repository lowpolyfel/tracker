
# flujo de arranque
# pantalla esperando camara usb
# si falla muestra error con opcion de reintentar
# si funciona abre preview con overlay esperando objetivo

import sys
from monitor.ui.startup_screen import show_waiting_for_camera, show_error_retry, show_camera_preview
from monitor.devices.camera import find_camera_index, open_camera


def main():
    while 1:
        show_waiting_for_camera()

        cam_index = find_camera_index(max_index=5)
        if cam_index < 0:
            action = show_error_retry(
                error_text="no se detecto camara usb",
                instructions="pulsa r para reintentar o q para salir"
            )
            if action == "retry":
                continue
            sys.exit(1)

        cap = open_camera(cam_index)
        if cap is None:
            action = show_error_retry(
                error_text=f"no se pudo abrir la camara en el indice {cam_index}",
                instructions="pulsa r para probar de nuevo o q para salir"
            )
            if action == "retry":
                continue
            sys.exit(2)

        next_action = show_camera_preview(cap, overlay_text="esperando objetivo... (q para salir)")
        if next_action == "back":
            continue
        break


if __name__ == "__main__":
    main()

