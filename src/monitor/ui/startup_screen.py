# pantallas basadas en opencv con estilo minimalista
# fondo claro, tipografia simple, alineado centrado

import cv2
import numpy as np

window_name = "wirebonder monitor"

# 0 es font hershey simplex en opencv
font = 0


def _render_text_canvas(title: str, subtitle: str = "", w: int = 960, h: int = 540):
    # fondo claro
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (245, 246, 248)

    # caja sutil
    cv2.rectangle(canvas, (24, 24), (w - 24, h - 24), (225, 227, 230), 2)

    # estilos de texto
    title_scale = 1.1
    subtitle_scale = 0.9
    title_color = (30, 30, 30)
    subtitle_color = (90, 95, 100)

    # calculo de centrado para el titulo
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, 2)
    tx = (w - tw) // 2
    ty = h // 2 - 20
    cv2.putText(canvas, title, (tx, ty), font, title_scale, title_color, 2, 16)

    if subtitle:
        (sw, sh), _ = cv2.getTextSize(subtitle, font, subtitle_scale, 2)
        sx = (w - sw) // 2
        sy = ty + 40
        cv2.putText(canvas, subtitle, (sx, sy), font, subtitle_scale, subtitle_color, 2, 16)

    return canvas


def show_waiting_for_camera():
    # pantalla informando busqueda de camara
    canvas = _render_text_canvas(
        title="buscando camara usb...",
        subtitle="conecta una camara. si se detecta, esta pantalla cambiara sola."
    )
    cv2.imshow(window_name, canvas)
    cv2.waitKey(250)


def show_error_retry(error_text: str, instructions: str = "pulsa r para reintentar, q para salir"):
    # pantalla de error con opciones
    action_color = (20, 120, 255)
    while 1:
        canvas = _render_text_canvas(title="error", subtitle=error_text)
        (aw, ah), _ = cv2.getTextSize(instructions, font, 0.9, 2)
        ax = (canvas.shape[1] - aw) // 2
        ay = (canvas.shape[0] // 2) + 80
        cv2.putText(canvas, instructions, (ax, ay), font, 0.9, action_color, 2, 16)

        cv2.imshow(window_name, canvas)
        k = cv2.waitKey(120) & 0xff
        if k in (ord("r"),):
            return "retry"
        if k in (ord("q"), 27):
            return "quit"


def show_camera_preview(cap, overlay_text: str = "esperando objetivo... (q para salir)"):
    # muestra el feed de camara con overlay
    badge_bg = (255, 255, 255)
    badge_border = (225, 227, 230)
    badge_text = (60, 65, 70)

    while 1:
        ok, frame = cap.read()
        if not ok:
            return "back"

        frame = _resize_keep_ratio(frame, target_w=960, target_h=540)

        # badge minimalista arriba a la izquierda
        pad = 16
        (tw, th), _ = cv2.getTextSize(overlay_text, font, 0.8, 2)
        box_w = tw + 24
        box_h = th + 20
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), badge_bg, -1)
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), badge_border, 2)
        cv2.putText(frame, overlay_text, (pad + 12, pad + box_h - 10), font, 0.8, badge_text, 2, 16)

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xff
        if k in (ord("q"), 27):
            return "quit"
        if k in (ord("b"),):
            return "back"


def _resize_keep_ratio(img, target_w=960, target_h=540):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), 3)  # 3 = inter_area
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (245, 246, 248)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas

