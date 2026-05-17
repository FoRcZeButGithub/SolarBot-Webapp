"""
app.py — HELIOS COMMAND: Flask backend for solar panel dust analysis.

Provides:
  GET  /camera          — Renders the Camera Vision page (camera.html)
  GET  /video_feed      — MJPEG stream of the active camera
  POST /switch_camera   — Switch cv2.VideoCapture index dynamically
  GET  /dust_status     — Returns latest dust score as JSON

Note (Windows): cv2.VideoCapture defaults to MSMF which causes grab errors
on some webcams. open_camera() tries CAP_DSHOW (DirectShow) first.
"""

import cv2
import sys
import platform
import threading
import time
import json
import os
from flask import Flask, Response, render_template, request, jsonify, send_from_directory

# ─────────────────────────────────────────────────────────────
# Flask App Initialisation
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


# ─────────────────────────────────────────────────────────────
# Global Camera State
# ─────────────────────────────────────────────────────────────
camera_lock   = threading.Lock()   # Thread-safe camera access
current_cam   = None               # Active cv2.VideoCapture object
current_index = 0                  # Current camera index (0, 1, 2 …)

# Latest dust analysis result (updated by background thread)
dust_state = {
    "score":  0.0,      # 0.0 (clean) → 100.0 (very dirty)
    "level":  "CLEAN",  # CLEAN | LOW | MODERATE | HIGH | CRITICAL
    "color":  "#00dbe7" # Colour matching the severity level
}

# Mapping score ranges to severity labels and colours
DUST_LEVELS = [
    (80, "CRITICAL", "#ff4444"),
    (60, "HIGH",     "#ff8800"),
    (40, "MODERATE", "#ffcc00"),
    (20, "LOW",      "#aaff00"),
    (0,  "CLEAN",    "#00dbe7"),
]


# ─────────────────────────────────────────────────────────────
# Dust Analysis Logic
# ─────────────────────────────────────────────────────────────
def calculate_dust_score(frame):
    """
    Estimate dust accumulation on a solar panel from a camera frame.

    Algorithm:
      1. Convert to grayscale.
      2. Apply Gaussian blur to suppress noise.
      3. Compute variance of the Laplacian — a measure of image sharpness.
         Dusty panels tend to have lower contrast / blurry patches.
      4. Analyse mean brightness; uniformly dim panels suggest heavy soiling.
      5. Combine both metrics into a 0–100 score where higher = dirtier.

    Returns:
      float — dust score in the range [0, 100]
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Sharpness metric ---
    # High variance → sharp (clean); low variance → blurry (possibly dusty)
    laplacian_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
    # Normalise: treat 0 as dirtiest, 500+ as cleanest; clamp to [0,1]
    sharpness_score = min(laplacian_var / 500.0, 1.0)

    # --- Brightness uniformity metric ---
    mean_brightness = blurred.mean() / 255.0  # 0 (dark/dirty) → 1 (bright/clean)

    # Combine (sharpness weighted higher)
    # clean_ratio → 1 means clean, 0 means dirty
    clean_ratio = (sharpness_score * 0.7) + (mean_brightness * 0.3)
    dust_score  = round((1.0 - clean_ratio) * 100, 1)

    return max(0.0, min(dust_score, 100.0))


def score_to_level(score):
    """Map a numeric dust score to a (label, colour) tuple."""
    for threshold, label, colour in DUST_LEVELS:
        if score >= threshold:
            return label, colour
    return "CLEAN", "#00dbe7"


# ─────────────────────────────────────────────────────────────
# Camera Management
# ─────────────────────────────────────────────────────────────
def open_camera(index):
    """
    Open cv2.VideoCapture at the given index.
    Tries CAP_DSHOW on Windows first (avoids MSMF errors), then default.
    Does a 3-frame warm-up read to confirm the camera actually delivers frames.
    """
    global current_cam, current_index

    if current_cam is not None and current_cam.isOpened():
        current_cam.release()
        current_cam = None

    is_windows = platform.system() == 'Windows'
    backends = [cv2.CAP_DSHOW] if is_windows else []
    backends.append(cv2.CAP_ANY)  # fallback to OS default

    for backend in backends:
        cap = cv2.VideoCapture(index, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        # Warm-up: try reading a few frames to verify real data flows
        success = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True
                break
        if success:
            current_cam   = cap
            current_index = index
            backend_name  = 'CAP_DSHOW' if backend == cv2.CAP_DSHOW else 'default'
            print(f"[CAM] Opened index {index} via {backend_name}")
            return cap
        cap.release()

    print(f"[CAM] Failed to open camera index {index}")
    current_cam = None
    return None


# Initialise with camera 0 on startup
with camera_lock:
    open_camera(0)


# ─────────────────────────────────────────────────────────────
# Background Dust Analysis Thread
# ─────────────────────────────────────────────────────────────
def dust_analysis_worker():
    """
    Runs continuously in a daemon thread.
    Reads the latest frame from the active camera every second,
    computes the dust score, and updates `dust_state`.
    """
    global dust_state
    while True:
        time.sleep(1)  # Analyse once per second (low overhead)
        with camera_lock:
            if current_cam is None or not current_cam.isOpened():
                continue
            ret, frame = current_cam.read()

        if ret and frame is not None:
            score         = calculate_dust_score(frame)
            label, colour = score_to_level(score)
            dust_state = {"score": score, "level": label, "color": colour}


# Start background thread as daemon (dies with main process)
analysis_thread = threading.Thread(target=dust_analysis_worker, daemon=True)
analysis_thread.start()


# ─────────────────────────────────────────────────────────────
# MJPEG Frame Generator
# ─────────────────────────────────────────────────────────────
def _make_placeholder_frame(message="NO CAMERA"):
    """
    Generate a black 640x360 JPEG frame with a status message.
    Returned as bytes so generate_frames() can yield it properly
    instead of yielding nothing (which triggers onerror in the browser).
    """
    import numpy as np
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Dim grid lines for aesthetics
    for x in range(0, w, 40):
        img[:, x] = [15, 15, 15]
    for y in range(0, h, 40):
        img[y, :] = [15, 15, 15]
    # Icon-style cross
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 30, cy - 30), (cx + 30, cy + 30), (60, 60, 80), 3, cv2.LINE_AA)
    cv2.line(img, (cx + 30, cy - 30), (cx - 30, cy + 30), (60, 60, 80), 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 50, (40, 40, 60), 2, cv2.LINE_AA)
    # Text
    cv2.putText(img, message, (cx - 120, cy + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 120), 2, cv2.LINE_AA)
    cv2.putText(img, "Check camera index or permissions",
                (cx - 170, cy + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 90), 1, cv2.LINE_AA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def generate_frames():
    """
    Generator that yields MJPEG-encoded frames from the active camera.
    Each frame is annotated with the live dust score overlay before encoding.
    """
    while True:
        with camera_lock:
            cam_ok = current_cam is not None and current_cam.isOpened()
            if cam_ok:
                ret, frame = current_cam.read()

        if not cam_ok:
            # Yield placeholder so the <img> stays alive (no onerror)
            placeholder = _make_placeholder_frame(f"NO CAMERA — index {current_index}")
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.5)  # slow poll when idle
            continue

        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # ── Draw dust score overlay on the frame ──────────────────────
        score  = dust_state["score"]
        level  = dust_state["level"]

        # Colour mapping (BGR for OpenCV)
        colour_map = {
            "CRITICAL": (68, 68, 255),
            "HIGH":     (0, 136, 255),
            "MODERATE": (0, 204, 255),
            "LOW":      (0, 255, 170),
            "CLEAN":    (231, 219, 0),
        }
        bgr = colour_map.get(level, (231, 219, 0))

        # Semi-transparent banner at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Dust score text
        text = f"DUST: {score:.1f}%  [{level}]  CAM:{current_index}"
        cv2.putText(
            frame, text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, bgr, 2, cv2.LINE_AA
        )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        # Yield as multipart MJPEG chunk
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )

        time.sleep(1 / 30)  # ~30 fps cap


# ─────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────

# Root: serve index.html (the main HELIOS COMMAND dashboard)
@app.route('/')
def index():
    """Serve the main dashboard page."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, 'index.html')


@app.route('/camera')
def camera_page():
    """Render the Camera Vision page."""
    return render_template('camera.html', current_index=current_index)


@app.route('/list_cameras')
def list_cameras():
    """
    Scan camera indices 0–5 and return the ones that are actually usable.
    Response: { "cameras": [0, 1], "current": 0 }
    """
    available = []
    is_windows = platform.system() == 'Windows'
    for idx in range(6):
        # Don't disturb the currently active camera
        if idx == current_index and current_cam is not None and current_cam.isOpened():
            available.append(idx)
            continue
        # Try to open briefly
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
        cap.release()
    return jsonify({"cameras": available, "current": current_index})


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint — used as the <img> src in the frontend."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    """
    Switch the active camera index.

    Request body (JSON): { "index": 1 }
    Response (JSON):     { "success": true, "index": 1 }
                      or { "success": false, "error": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    new_index = data.get('index')

    if new_index is None or not isinstance(new_index, int) or new_index < 0:
        return jsonify({"success": False, "error": "Invalid camera index"}), 400

    with camera_lock:
        cap = open_camera(new_index)

    if cap is not None:
        return jsonify({"success": True, "index": new_index})
    else:
        return jsonify({"success": False, "error": f"Camera {new_index} not available"}), 503


@app.route('/dust_status')
def dust_status():
    """Return the latest dust analysis result as JSON."""
    return jsonify({
        **dust_state,
        "camera_index": current_index
    })


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PORT = 5501
    # debug=False is important - the camera thread must not be forked twice
    print(f"\n[HELIOS] Dashboard     -> http://localhost:{PORT}/")
    print(f"[HELIOS] Camera Vision -> http://localhost:{PORT}/camera\n")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
