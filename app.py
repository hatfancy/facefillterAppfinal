import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import os
import threading

app = Flask(__name__)

# ── Face Detection (Haar Cascade with tuned parameters) ─────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Also load the alt cascade for better detection
face_cascade_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

def detect_faces(frame):
    """Try multiple detection strategies for robust face finding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for detection

    # Try primary cascade with relaxed settings
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If primary found nothing, try alt cascade
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    return faces

# ── Load emoji assets ───────────────────────────────────────────────────────
def load_alpha_image(path):
    if not os.path.exists(path):
        print(f"Warning: Asset not found at {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        print(f"Loaded asset: {path}  shape={img.shape}")
    return img

assets = {
    "smiley": load_alpha_image("assets/smiley.png"),
    "wink":   load_alpha_image("assets/wink.png"),
    "cool":   load_alpha_image("assets/cool.png"),
}

# ── Shared state (thread-safe) ──────────────────────────────────────────────
current_filter = "none"
filter_lock = threading.Lock()

# ── Image overlay helper ────────────────────────────────────────────────────
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # If overlay has no alpha channel, just paste it
    if overlay.shape[2] == 3:
        y1, y2 = max(0, y), min(bg_h, y + ov_h)
        x1, x2 = max(0, x), min(bg_w, x + ov_w)
        ov_y1 = max(0, -y)
        ov_x1 = max(0, -x)
        h = y2 - y1
        w = x2 - x1
        if h > 0 and w > 0:
            background[y1:y2, x1:x2] = overlay[ov_y1:ov_y1+h, ov_x1:ov_x1+w]
        return background

    # With alpha channel
    y1, y2 = max(0, y), min(bg_h, y + ov_h)
    x1, x2 = max(0, x), min(bg_w, x + ov_w)
    ov_y1, ov_y2 = max(0, -y), min(ov_h, bg_h - y)
    ov_x1, ov_x2 = max(0, -x), min(ov_w, bg_w - x)

    if ov_y1 >= ov_y2 or ov_x1 >= ov_x2:
        return background

    alpha_s = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha_s * overlay[ov_y1:ov_y2, ov_x1:ov_x2, c]
            + alpha_l * background[y1:y2, x1:x2, c]
        )
    return background

# ── MJPEG frame generator ──────────────────────────────────────────────────
def generate_frames():
    global current_filter

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened successfully. Streaming...")

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Read the current filter in a thread-safe way
        with filter_lock:
            active_filter = current_filter

        if active_filter != "none" and assets.get(active_filter) is not None:
            faces = detect_faces(frame)

            for (x, y, fw, fh) in faces:
                filter_img = assets[active_filter]
                if filter_img is not None:
                    scale = 1.5
                    scaled_w = int(fw * scale)
                    scaled_h = int(fh * scale)
                    if scaled_w > 0 and scaled_h > 0:
                        resized = cv2.resize(filter_img, (scaled_w, scaled_h))
                        off_x = x - int((scaled_w - fw) / 2)
                        off_y = y - int((scaled_h - fh) / 2)
                        frame = overlay_transparent(frame, resized, off_x, off_y)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
        )

    camera.release()

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/set_filter', methods=['POST'])
def set_filter():
    global current_filter
    data = request.get_json()
    new_filter = data.get('filter', 'none')
    with filter_lock:
        current_filter = new_filter
    print(f"Filter changed to: {new_filter}")
    return jsonify({"status": "ok", "filter": new_filter})

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
