import os
import queue
import requests
import threading
import time

import cv2
from ultralytics import YOLO
from collections import Counter

PERSONA = 0
GATO = 15
PERRO = 16
CLASSES = [PERSONA, GATO, PERRO]
DEVICE = "mps"
CONFIDENCE = 0.50
EXIT_IN_SECONDS = 3
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK"]

cola = queue.Queue(maxsize=10)

def enviar(texto, jpg_bytes=None):
    try:

        if jpg_bytes:
            requests.post(
                WEBHOOK_URL,
                data={"content": texto},
                files={"file": ("foto.jpg", jpg_bytes, "image/jpeg")},
                timeout=10,
            )
        else:
            requests.post(WEBHOOK_URL, data={"content": texto}, timeout=10)
    except requests.RequestException as e:
        print(f"Error sending message: {e}")

def worker():
    while True:
        texto, jpg = cola.get()
        enviar(texto, jpg)
        cola.task_done()

threading.Thread(target=worker, daemon=True).start()

def alert(texto, frame=None):
    jpg = None
    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            jpg = buf.tobytes()
    try:
        cola.put_nowait((texto, jpg))
    except queue.Full:
        print("Queue is full, discarding message")


# --- Main program ---

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

present = False
last_seen = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, classes=CLASSES, conf=CONFIDENCE ,device=DEVICE, verbose=False)[0]
    out = r.plot()
   # n = len(r.boxes)
    names = [model.names[int(c)] for c in r.boxes.cls]
    total = Counter(names) # for example ['person': 2, 'dog': 1]
    texto = ",".join(f"{k}: {v}" for k,v in total.items()) or "nothing"
    now = time.time()

    if len(r.boxes) > 0:
        last_seen = now
        if not present:
            present = True
            print("person detected: ", texto)
            alert(f"person detected: {texto}", out)
    elif present and now - last_seen > EXIT_IN_SECONDS:
        present = False
        print("person left: ", texto)
        alert(f"person left: {texto}")    

    cv2.putText(out, texto, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("cam", out)
    if cv2.waitKey(1)==27: # press esc to exit
        break

cap.release()
cv2.destroyAllWindows()
               