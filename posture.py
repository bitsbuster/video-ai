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
STAND_RATIO = 0.62   # knee this far below the hip (in torso lengths) -> standing
SIT_RATIO = 0.50     # knee this close to the hip height -> sitting
LYING_RATIO = 0.90   # box width / height at or above this -> lying
WINDOW = 15          # frames used to smooth the state
MIN_VOTES = 10       # frames of the window that must agree


def _mid(xy, conf, ids):
    pts = [xy[i] for i in ids if conf[i] >= KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def classify(xy, conf, box_ratio=None):
    """Return (state, thigh_drop). state: standing, sitting, lying or unknown.

    box_ratio is the width / height of the person box. A box wider than tall
    means lying, and it does not depend on the keypoints.

    thigh_drop = how far the knees are below the hips, divided by the torso length.
    Standing is about 1.0 (the thigh is vertical). Sitting is close to 0 (horizontal).
    """
    if box_ratio is not None and box_ratio >= LYING_RATIO:
        return "lying", None
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


def features(xy, conf, box_h):
    """Debug text: torso and head position, as a fraction of the box height.

    tx, ty: horizontal and vertical distance from shoulders to hips.
    head: nose height above the hips (positive = nose is higher in the image).
    """
    shoulder = _mid(xy, conf, SHOULDERS)
    hip = _mid(xy, conf, HIPS)
    if shoulder is None or hip is None or box_h <= 0:
        return "-"
    tx = abs(hip[0] - shoulder[0]) / box_h
    ty = abs(hip[1] - shoulder[1]) / box_h
    head = "-" if conf[0] < KP_CONF else f"{(hip[1] - xy[0][1]) / box_h:.2f}"
    return f"tx={tx:.2f} ty={ty:.2f} head={head}"


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

        state, drop, seen, ratio, feat = "none", None, "-", "-", "-"
        if len(r.boxes) > 0:
            # use the biggest person in the image
            areas = (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy()
            i = int(np.argmax(areas))
            bw, bh = r.boxes.xywh[i, 2:].tolist()
            box_ratio = bw / bh  # width / height of the box
            ratio = f"{box_ratio:.2f}"
            xy = r.keypoints.xy[i].cpu().numpy()
            conf = r.keypoints.conf[i].cpu().numpy()
            state, drop = classify(xy, conf, box_ratio)
            feat = features(xy, conf, bh)
            # how many shoulders / hips / knees are visible (0-2 each)
            seen = "".join(f"{n}{sum(conf[j] >= KP_CONF for j in ids)}"
                           for n, ids in (("S", SHOULDERS), ("H", HIPS), ("K", KNEES)))
        history.append(state)

        now = time.time()
        if now - last_print >= 1:
            last_print = now
            value = "-" if drop is None else f"{drop:.2f}"
            print(f"raw={state} thigh_drop={value} seen={seen} box_w/h={ratio} {feat} stable={current}", flush=True)

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
