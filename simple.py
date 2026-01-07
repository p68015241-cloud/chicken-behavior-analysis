import time
import threading
from datetime import datetime
from collections import deque, defaultdict

import os
import json

import cv2
import numpy as np
from ultralytics import YOLO

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "model/best.pt"
VIDEO_SOURCE = 0
PROCESS_RES = (640, 640)

CONF = 0.25
IOU = 0.50


STATIONARY_SPEED = 3.0
LETHARGY_FRAMES = 150  

SPEED_WINDOW = 30               
CALIBRATION_SAMPLES = 500       
CONTAMINATION = 0.02            # expected anomaly rate (2%)
PANIC_MIN_SPEED = 10.0          
PANIC_COOLDOWN_SEC = 10.0       

JPEG_QUALITY = 80

LOG_DIR = "anomaly_logs"
EVENT_LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

latest_jpeg = None
events = deque(maxlen=200)
lock = threading.Lock()

# Tracking states
prev_pos = {}  # tid -> (cx, cy)
speed_hist = defaultdict(lambda: deque(maxlen=SPEED_WINDOW))
stationary_counter = defaultdict(int)
last_panic_time = defaultdict(float)

# IForest states
scaler = StandardScaler()
iforest = IsolationForest(contamination=CONTAMINATION, random_state=42)
trained = False
train_buffer = []

def speed_features(speeds: deque) -> np.ndarray | None:
    if len(speeds) < SPEED_WINDOW:
        return None
    v = np.array(speeds, dtype=np.float32)
    return np.array([
        v.mean(),
        v.std(),
        v.max(),
    ], dtype=np.float32)


def log_event(event: dict):
    with lock:
        events.appendleft(event)

def save_screenshot(frame, event_type, track_id, extra=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra_part = ""
    if extra is not None:
        extra_part = f"_{extra}".replace(" ", "_").replace("/", "_")

    filename = os.path.join(LOG_DIR, f"{event_type}_id{track_id}_{ts}{extra_part}.jpg")
    cv2.imwrite(filename, frame)
    return filename

# Adding event to the json file
def append_event_to_file(event_dict):
    with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")


def camera_loop():
    global latest_jpeg, trained, train_buffer

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    prev_time = time.time()

    while True:
        ok, raw = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.resize(raw, PROCESS_RES)
        now = time.time()
        dt = max(now - prev_time, 1e-6)
        prev_time = now

        # YOLO tracking
        results = model.track(frame, conf=CONF, iou=IOU, persist=True, verbose=False)[0]

        # Overlay that shows IForest status
        status_text = "CALIBRATING..." if not trained else "RUNNING"
        cv2.putText(frame, status_text, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # if there are any detections
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)

            # Collecting features, speed history and lethargy counters
            feat_vectors = []
            feat_ids = []

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # compute speed
                speed = 0.0
                if tid in prev_pos:
                    px, py = prev_pos[tid]
                    speed = float(np.hypot(cx - px, cy - py) / dt)

                prev_pos[tid] = (cx, cy)
                speed_hist[tid].append(speed)

                # lethargy rule
                if speed < STATIONARY_SPEED:
                    stationary_counter[tid] += 1
                else:
                    stationary_counter[tid] = 0

                # build features if possible
                feats = speed_features(speed_hist[tid])
                if feats is not None:
                    feat_vectors.append(feats)
                    feat_ids.append(tid)

            # Train IsolationForest after calibration_samples collected
            if not trained:
                if feat_vectors:
                    train_buffer.extend(feat_vectors)
                if len(train_buffer) >= CALIBRATION_SAMPLES:
                    X = np.vstack(train_buffer)
                    Xs = scaler.fit_transform(X)
                    iforest.fit(Xs)
                    trained = True
                    train_buffer = []

                    event = {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "type": "SYSTEM",
                        "message": "IsolationForest calibrated and running"
                    }
                    log_event(event)
                    append_event_to_file(event)

            # PANIC detection (IsolationForest)
            panic_ids = set()
            if trained and feat_vectors:
                X = np.vstack(feat_vectors)
                Xs = scaler.transform(X)
                preds = iforest.predict(Xs)  # -1 anomaly

                for tid, pred in zip(feat_ids, preds):
                    avg_speed = float(np.mean(speed_hist[tid])) if len(speed_hist[tid]) else 0.0
                    if avg_speed < PANIC_MIN_SPEED:
                        continue

                    t_now = time.time()
                    if pred == -1 and (t_now - last_panic_time[tid]) > PANIC_COOLDOWN_SEC:
                        last_panic_time[tid] = t_now
                        panic_ids.add(tid)

                        event = {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "type": "PANIC",
                            "track_id": int(tid),
                            "avg_speed": avg_speed,
                        }
                        path = save_screenshot(frame, "PANIC", tid, extra=f"avg{avg_speed:.1f}")
                        event["file"] = path

                        log_event(event)
                        append_event_to_file(event)

            # Second pass: draw boxes + log lethargy when it first triggers
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                is_leth = stationary_counter[tid] > LETHARGY_FRAMES
                is_panic = tid in panic_ids

                if is_leth:
                    color = (0, 0, 255)
                    label = "LETHARGY"
                elif is_panic:
                    color = (0, 165, 255)
                    label = "PANIC"
                else:
                    color = (0, 255, 0)
                    label = f"ID {tid}"

                # Logging the lethargy
                if is_leth and stationary_counter[tid] == LETHARGY_FRAMES + 1:
                    event = {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "type": "LETHARGY",
                        "track_id": int(tid),
                    }
                    path = save_screenshot(frame, "LETHARGY", tid)
                    event["file"] = path

                    log_event(event)
                    append_event_to_file(event)

                speed_now = speed_hist[tid][-1] if len(speed_hist[tid]) else 0.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} v:{speed_now:.1f}",
                            (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Streaming with jpg
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()

# Server
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def start_thread():
    threading.Thread(target=camera_loop, daemon=True).start()

def mjpeg_gen():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with lock:
        return JSONResponse(list(events))
# Run with: uvicorn simple:app --host [ip] --port [port]