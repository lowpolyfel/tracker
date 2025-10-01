# pantallas basadas en opencv y pillow estilo minimalista
# soporta overlay opcional para dibujar detecciones y badges de estado

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

window_name = "wirebonder monitor"


def _render_text_canvas(title: str, subtitle: str = "", w: int = 960, h: int = 540):
    img = Image.new("RGB", (w, h), (245, 246, 248))
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    tw, th = draw.textbbox((0, 0), title, font=font_title)[2:]
    draw.text(((w - tw) // 2, h // 2 - 40), title, font=font_title, fill=(30, 30, 30))

    if subtitle:
        sw, sh = draw.textbbox((0, 0), subtitle, font=font_sub)[2:]
        draw.text(((w - sw) // 2, h // 2 + 10), subtitle, font=font_sub, fill=(90, 90, 90))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def show_waiting_for_camera():
    canvas = _render_text_canvas(
        title="buscando camara usb...",
        subtitle="conecta una camara; si se detecta, esta pantalla cambiara sola"
    )
    cv2.imshow(window_name, canvas)
    cv2.waitKey(250)


def show_error_retry(error_text: str, instructions: str = "pulsa r para reintentar, q para salir"):
    action_text = instructions
    while 1:
        canvas = _render_text_canvas(title="error", subtitle=error_text)
        (tw, th), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        ax = (canvas.shape[1] - tw) // 2
        ay = (canvas.shape[0] // 2) + 80
        cv2.putText(canvas, action_text, (ax, ay), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 120, 255), 2, 16)
        cv2.imshow(window_name, canvas)
        k = cv2.waitKey(120) & 0xff
        if k in (ord("r"),):
            return "retry"
        if k in (ord("q"), 27):
            return "quit"


def show_camera_preview(cap, overlay_text: str = "esperando objetivo... (q para salir)", overlay_fn=None):
    badge_bg = (255, 255, 255)
    badge_border = (225, 227, 230)
    badge_text = (60, 65, 70)

    while 1:
        ok, frame = cap.read()
        if not ok:
            return "back"

        frame = _resize_keep_ratio(frame, target_w=960, target_h=540)

        # badge superior
        pad = 16
        (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        box_w = tw + 24
        box_h = th + 20
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), badge_bg, -1)
        cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), badge_border, 2)
        cv2.putText(frame, overlay_text, (pad + 12, pad + box_h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, badge_text, 2, 16)

        # overlay de deteccion (opcional)
        if overlay_fn is not None:
            try:
                overlay_fn(frame)
            except Exception:
                pass

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xff
        if k in (ord("q"), 27):
            return "quit"
        if k in (ord("b"),):
            return "back"


def draw_detection_overlay(frame, state: dict, conf_min: float = 0.25):
    # dibuja cajas y badges para reel y tip si existen
    h, w = frame.shape[:2]

    def _box(item, color, label):
        if not item or item.get("conf", 0.0) < conf_min:
            return
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {item['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 12, y1), (255, 255, 255), -1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 12, y1), (225, 227, 230), 1)
        cv2.putText(frame, txt, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 1, 16)

    _box(state.get("reel"), (80, 200, 120), "reel")
    _box(state.get("tip"), (80, 120, 220), "tip")

    # indicadores de estado en la esquina
    x0, y0 = w - 200, 16
    reel_ok = state.get("reel") is not None and state["reel"]["conf"] >= conf_min
    tip_ok = state.get("tip") is not None and state["tip"]["conf"] >= conf_min

    def _pill(y, txt, ok):
        bg = (235, 245, 238) if ok else (245, 236, 236)
        bd = (180, 230, 190) if ok else (232, 196, 196)
        fg = (40, 120, 60) if ok else (140, 70, 70)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, y), (x0 + tw + 24, y + th + 14), bg, -1)
        cv2.rectangle(frame, (x0, y), (x0 + tw + 24, y + th + 14), bd, 2)
        cv2.putText(frame, txt, (x0 + 12, y + th + 2), cv2.FONT_HERSHEY_DUPLEX, 0.6, fg, 1, 16)

    _pill(y0, "reel", reel_ok)
    _pill(y0 + 40, "tip", tip_ok)


def _resize_keep_ratio(img, target_w=960, target_h=540):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (245, 246, 248)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_tracking_overlay(frame, tracking_state: dict):
    # pinta cajas del tracker en colores distintos a deteccion
    color_tip = (60, 200, 255)
    color_reel = (120, 220, 120)

    def _draw(b, color, label):
        if not b:
            return
        x1, y1, x2, y2 = b
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, 16)

    tip = tracking_state.get("tip", {})
    reel = tracking_state.get("reel", {})
    _draw(tip.get("bbox"), color_tip, "tip")
    _draw(reel.get("bbox"), color_reel, "reel")
