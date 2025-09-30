# arranque del programa
# pantalla inicial buscando camara usb
# si no hay camara mostrar error y pedir reintento
# si hay camara abrir preview con mensaje esperando objetivo

import sys
from monitor.ui.startup_screen import show_waiting_for_camera, show_error_retry, show_camera_preview
from monitor.devices.camera import find_camera_index, open_camera


def main():
    while True:
        show_waiting_for_camera()

        cam_index = find_camera_index(max_index=5)
        if cam_index is None:
            action = show_error_retry(
                error_text="no se detecto camara usb",
                instructions="pulsa r para reintentar o q para salir"
            )
            if action == "retry":
                continue
            else:
                sys.exit(1)
        else:
            cap = open_camera(cam_index)
            if cap is None:
                action = show_error_retry(
                    error_text=f"no se pudo abrir la camara en el indice {cam_index}",
                    instructions="pulsa r para probar de nuevo o q para salir"
                )
                if action == "retry":
                    continue
                else:
                    sys.exit(2)

            next_action = show_camera_preview(cap, overlay_text="esperando objetivo... (q para salir)")
            if next_action == "back":
                continue
            else:
                break


if __name__ == "__main__":
    main()

