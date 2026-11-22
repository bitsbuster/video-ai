"""Experimental: detect if the person is standing or sitting using YOLO pose.

Run it instead of detect.py (stop the main container first, so the camera
has a free RTSP session):

    docker compose stop
    docker compose run --rm video-ai python3 posture.py

It prints the state every second, with the measured value, so you can tune
the thresholds. With ALERT=1 it also sends the state changes to Discord.
"""
import os
import threading
import time
from collections import Counter, deque

import numpy as np

# COCO keypoints used
SHOULDERS = (5, 6)
HIPS = (11, 12)
KNEES = (13, 14)

KP_CONF = 0.3        # minimum confidence of a keypoint
STAND_RATIO = 0.70   # knee this far below the hip (in torso lengths) -> standing
SIT_RATIO = 0.45     # knee this close to the hip height -> sitting
WINDOW = 15          # frames used to smooth the state
MIN_VOTES = 10       # frames of the window that must agree


def _mid(xy, conf, ids):
    pts = [xy[i] for i in ids if conf[i] >= KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def classify(xy, conf):
    """Return (state, thigh_drop). state: standing, sitting, lying or unknown.

    thigh_drop = how far the knees are below the hips, divided by the torso length.
    Standing is about 1.0 (the thigh is vertical). Sitting is close to 0 (horizontal).
    """
    shoulder = _mid(xy, conf, SHOULDERS)
    hip = _mid(xy, conf, HIPS)
    knee = _mid(xy, conf, KNEES)
    if shoulder is None or hip is None:
        return "unknown", None
    torso = float(np.linalg.norm(hip - shoulder))
    if torso < 1:
        return "unknown", None
    if abs(hip[0] - shoulder[0]) > abs(hip[1] - shoulder[1]):
        return "lying", None  # the torso is more horizontal than vertical
    if knee is None:
        return "unknown", None
    drop = float((knee[1] - hip[1]) / torso)
    if drop >= STAND_RATIO:
        return "standing", drop
    if drop <= SIT_RATIO:
        return "sitting", drop
    return "unknown", drop


def main():
    import cv2
    import requests
    from ultralytics import YOLO

    device = os.environ.get("DEVICE", "cuda")
    camera = os.environ.get("CAMERA", "0")
    camera = int(camera) if camera.isdigit() else camera
    imgsz = int(os.environ.get("IMGSZ", "1280"))  # a bigger size helps when the person is far
    webhook = os.environ.get("DISCORD_WEBHOOK") if os.environ.get("ALERT") == "1" else None

    def send(text, jpg):
        try:
            requests.post(webhook, data={"content": text},
                          files={"file": ("foto.jpg", jpg, "image/jpeg")}, timeout=10)
        except requests.RequestException as e:
            print(f"Error sending message: {e}")

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera. Check CAMERA.")

    history = deque(maxlen=WINDOW)
    current = None
    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, imgsz=imgsz, conf=0.3, device=device, verbose=False)[0]

        state, drop = "none", None
        if len(r.boxes) > 0:
            # use the biggest person in the image
            areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            i = int(np.argmax(areas))
            xy = r.keypoints.xy[i].cpu().numpy()
            conf = r.keypoints.conf[i].cpu().numpy()
            state, drop = classify(xy, conf)
        history.append(state)

        now = time.time()
        if now - last_print >= 1:
            last_print = now
            value = "-" if drop is None else f"{drop:.2f}"
            print(f"raw={state} thigh_drop={value} stable={current}", flush=True)

        top, votes = Counter(history).most_common(1)[0]
        if votes >= MIN_VOTES:
            if top == "none":
                current = None
            elif top != "unknown" and top != current:
                current = top
                print(f"STATE CHANGE: {current}", flush=True)
                if webhook:
                    ok_jpg, buf = cv2.imencode(".jpg", r.plot())
                    if ok_jpg:
                        threading.Thread(target=send, args=(f"person is {current}", buf.tobytes()),
                                         daemon=True).start()

    cap.release()


if __name__ == "__main__":
    main()
