# Wirebonder Monitor (Raspberry Pi 4, Debian)

Sistema de monitoreo para detectar actividad/idle de una wire bonder usando visión por computadora.
Arquitectura preparada para Python (src/), pruebas (tests), datos (data/), modelos (models/),
notebooks y despliegue en Raspberry Pi.

## Requisitos del sistema
- Raspberry Pi 4 (4GB recomendado) con Debian/Raspberry Pi OS.
- Python 3.11+ (o el que tenga tu sistema).
- OpenCV desde apt (`python3-opencv`) para mayor compatibilidad en ARM.
- ffmpeg (para manejo de video).

## Entorno
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-min.txt
# (Opcional y con cautela en ARM)
# pip install -r requirements-ml.txt
