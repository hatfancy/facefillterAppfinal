import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import os
import threading
import time
import base64
from datetime import datetime

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


# ── Generate emoji overlays programmatically ────────────────────────────────
def generate_emoji(emoji_type, size=256):
    """Create emoji-like RGBA overlay images using OpenCV drawing."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    c = size // 2  # center
    r = size // 2 - 12  # radius

    if emoji_type == "heart_eyes":
        # Yellow face
        cv2.circle(img, (c, c), r, (0, 220, 255, 255), -1)
        cv2.circle(img, (c, c), r, (0, 180, 210, 255), 3)
        # Heart-shaped left eye
        pts_l = np.array([
            [c - 55, c - 30], [c - 75, c - 55], [c - 85, c - 35],
            [c - 55, c - 5], [c - 25, c - 35], [c - 35, c - 55]
        ], np.int32)
        cv2.fillPoly(img, [pts_l], (60, 60, 220, 255))
        # Heart-shaped right eye
        pts_r = pts_l.copy()
        pts_r[:, 0] = 2 * c - pts_l[:, 0]
        cv2.fillPoly(img, [pts_r], (60, 60, 220, 255))
        # Big smile
        cv2.ellipse(img, (c, c + 35), (55, 40), 0, 10, 170, (0, 0, 180, 255), 4)

    elif emoji_type == "clown":
        # White face
        cv2.circle(img, (c, c), r, (240, 240, 245, 255), -1)
        cv2.circle(img, (c, c), r, (200, 200, 210, 255), 3)
        # Red nose
        cv2.circle(img, (c, c + 5), 25, (50, 50, 230, 255), -1)
        # Eyes
        cv2.circle(img, (c - 45, c - 30), 14, (120, 70, 20, 255), -1)
        cv2.circle(img, (c + 45, c - 30), 14, (120, 70, 20, 255), -1)
        # Big smile
        cv2.ellipse(img, (c, c + 40), (60, 35), 0, 5, 175, (50, 50, 230, 255), 5)
        # Rainbow hair — coloured arcs on top
        colours = [(0, 0, 255, 255), (0, 140, 255, 255), (0, 255, 255, 255),
                   (0, 255, 0, 255), (255, 0, 0, 255), (255, 0, 130, 255)]
        for i, col in enumerate(colours):
            cv2.ellipse(img, (c, c - 50), (r - 5 + i * 8, 55 + i * 6),
                        0, 190, 350, col, 5)

    elif emoji_type == "ghost":
        # Ghost body — white with wavy bottom
        pts = []
        # Top dome
        for angle in range(180, 361):
            rad = np.radians(angle)
            px = int(c + r * 0.7 * np.cos(rad))
            py = int(c - 15 + r * 0.7 * np.sin(rad))
            pts.append([px, py])
        # Wavy bottom
        bottom_y = c + int(r * 0.85)
        wave_pts = 40
        for i in range(wave_pts + 1):
            x = int(c - r * 0.7 + (2 * r * 0.7 * i / wave_pts))
            y = int(bottom_y + 18 * np.sin(i * np.pi * 3 / wave_pts))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255, 230))
        # Eyes
        cv2.circle(img, (c - 28, c - 15), 14, (60, 60, 60, 255), -1)
        cv2.circle(img, (c + 28, c - 15), 14, (60, 60, 60, 255), -1)
        # Mouth
        cv2.ellipse(img, (c, c + 25), (18, 14), 0, 0, 360, (60, 60, 60, 255), -1)

    elif emoji_type == "pumpkin":
        # Orange pumpkin body
        cv2.ellipse(img, (c, c + 10), (r - 5, r - 15), 0, 0, 360, (30, 130, 240, 255), -1)
        cv2.ellipse(img, (c, c + 10), (r - 5, r - 15), 0, 0, 360, (15, 100, 200, 255), 3)
        # Stem
        cv2.rectangle(img, (c - 10, c - r + 5), (c + 10, c - r + 30), (30, 140, 60, 255), -1)
        # Triangle eyes
        eye_l = np.array([[c - 55, c - 10], [c - 30, c - 40], [c - 5, c - 10]], np.int32)
        eye_r = np.array([[c + 5, c - 10], [c + 30, c - 40], [c + 55, c - 10]], np.int32)
        cv2.fillPoly(img, [eye_l], (20, 220, 255, 255))
        cv2.fillPoly(img, [eye_r], (20, 220, 255, 255))
        # Triangle nose
        nose = np.array([[c - 15, c + 5], [c, c - 15], [c + 15, c + 5]], np.int32)
        cv2.fillPoly(img, [nose], (20, 220, 255, 255))
        # Jagged mouth
        mouth_pts = []
        for i in range(9):
            x = c - 55 + i * 14
            y = c + 35 + (15 if i % 2 == 0 else -5)
            mouth_pts.append([x, y])
        mouth_pts = np.array(mouth_pts, np.int32)
        cv2.fillPoly(img, [mouth_pts], (20, 220, 255, 255))

    elif emoji_type == "devil":
        # Purple face
        cv2.circle(img, (c, c + 8), r - 10, (180, 50, 130, 255), -1)
        cv2.circle(img, (c, c + 8), r - 10, (140, 30, 100, 255), 3)
        # Horns
        horn_l = np.array([[c - 60, c - 55], [c - 80, c - r - 10], [c - 30, c - 70]], np.int32)
        horn_r = np.array([[c + 60, c - 55], [c + 80, c - r - 10], [c + 30, c - 70]], np.int32)
        cv2.fillPoly(img, [horn_l], (40, 40, 200, 255))
        cv2.fillPoly(img, [horn_r], (40, 40, 200, 255))
        # Evil eyes
        cv2.ellipse(img, (c - 40, c - 15), (18, 12), -15, 0, 360, (50, 255, 255, 255), -1)
        cv2.circle(img, (c - 40, c - 15), 7, (30, 30, 30, 255), -1)
        cv2.ellipse(img, (c + 40, c - 15), (18, 12), 15, 0, 360, (50, 255, 255, 255), 255)
        cv2.circle(img, (c + 40, c - 15), 7, (30, 30, 30, 255), -1)
        # Smirk
        cv2.ellipse(img, (c, c + 40), (45, 25), 0, 5, 175, (40, 40, 200, 255), 4)

    return img


# Load file-based + generated emojis into one dict
assets = {
    "smiley":     load_alpha_image("assets/smiley.png"),
    "wink":       load_alpha_image("assets/wink.png"),
    "cool":       load_alpha_image("assets/cool.png"),
    "heart_eyes": generate_emoji("heart_eyes"),
    "clown":      generate_emoji("clown"),
    "ghost":      generate_emoji("ghost"),
    "pumpkin":    generate_emoji("pumpkin"),
    "devil":      generate_emoji("devil"),
}

# ── Shared state (thread-safe) ──────────────────────────────────────────────
current_filter = "none"
filter_lock = threading.Lock()

# Store the latest frame for capture
latest_frame = None
frame_lock = threading.Lock()

# Gallery / Hub — in-memory post storage
gallery_posts = []
gallery_lock = threading.Lock()
post_id_counter = 0

# ── Low-Light Adaptive Enhancement ──────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def enhance_low_light(frame):
    """
    Detect dark frames and apply CLAHE + brightness boost.
    This dramatically improves face detection and visual quality
    in dim / poorly-lit environments.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # Only activate when the frame is genuinely dark (threshold ~80)
    if mean_brightness < 80:
        # Convert to LAB colour space – L channel is lightness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        l = clahe.apply(l)

        # Additional brightness boost proportional to how dark it is
        boost = int((80 - mean_brightness) * 0.5)
        l = cv2.add(l, boost)

        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame

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


# ═══════════════════════════════════════════════════════════════════════════
#  AI / CV-based creative filters  (existing 3 + new 5 + swirl)
# ═══════════════════════════════════════════════════════════════════════════

def apply_pixel_art(frame, faces):
    """
    Pixelate each detected face into large colourful blocks,
    giving an 8-bit retro video-game aesthetic.
    """
    for (x, y, fw, fh) in faces:
        pad = int(fw * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + fw + pad)
        y2 = min(frame.shape[0], y + fh + pad)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        h, w = face_roi.shape[:2]
        pixel_size = max(1, min(w, h) // 12)
        small = cv2.resize(face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                           interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(pixelated, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        pixelated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        frame[y1:y2, x1:x2] = pixelated
    return frame


def apply_neon_glow(frame, faces):
    """
    Surround each detected face with animated neon contour lines
    that pulse with vibrant colours (cyan, magenta, yellow cycle).
    """
    for (x, y, fw, fh) in faces:
        center_x = x + fw // 2
        center_y = y + fh // 2
        axes = (int(fw * 0.65), int(fh * 0.75))

        t = time.time()
        phase = t % 3.0
        if phase < 1.0:
            colour = (255, 255, 0)    # Cyan
        elif phase < 2.0:
            colour = (255, 0, 255)    # Magenta
        else:
            colour = (0, 255, 255)    # Yellow

        overlay = frame.copy()
        for i, thickness in enumerate([8, 5, 3, 2]):
            scale = 1.0 + i * 0.08
            scaled_axes = (int(axes[0] * scale), int(axes[1] * scale))
            cv2.ellipse(overlay, (center_x, center_y), scaled_axes,
                        0, 0, 360, colour, thickness, cv2.LINE_AA)

        alpha = 0.7 + 0.3 * np.sin(t * 4)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        glow_mask = np.zeros_like(frame)
        cv2.ellipse(glow_mask, (center_x, center_y),
                    (int(axes[0] * 1.1), int(axes[1] * 1.1)),
                    0, 0, 360, colour, 12, cv2.LINE_AA)
        glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 10)
        frame = cv2.add(frame, (glow_mask * 0.4).astype(np.uint8))

    return frame


def apply_sketch(frame, faces):
    """
    Transform the entire frame into a pencil-sketch / comic-book style
    with bold edges and reduced colour palette.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=2
    )

    colour = cv2.bilateralFilter(frame, d=9, sigmaColor=200, sigmaSpace=200)
    div = 48
    colour = (colour // div) * div + div // 2

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    sketch = cv2.bitwise_and(colour, edges_bgr)
    return sketch


# ─── NEW AI Effects ─────────────────────────────────────────────────────────

def apply_swirl(frame, faces):
    """
    Apply a mesmerising swirl distortion to each detected face.
    The swirl animates over time, rotating continuously.
    """
    h, w = frame.shape[:2]
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw // 2
        cy = fy + fh // 2
        radius = int(max(fw, fh) * 0.8)

        # Animate swirl angle over time
        t = time.time()
        swirl_strength = 2.5 * np.sin(t * 1.5)

        # Build coordinate maps
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Distance from face center
        dx = x_coords - cx
        dy = y_coords - cy
        dist = np.sqrt(dx**2 + dy**2)

        # Only swirl within the radius
        mask = dist < radius
        if not np.any(mask):
            continue

        # Normalised distance (0 at center, 1 at edge)
        norm_dist = np.where(mask, dist / radius, 1.0)

        # Swirl angle falls off towards the edges
        angle = swirl_strength * (1.0 - norm_dist) ** 2

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        new_x = np.where(mask, cx + dx * cos_a - dy * sin_a, x_coords).astype(np.float32)
        new_y = np.where(mask, cy + dx * sin_a + dy * cos_a, y_coords).astype(np.float32)

        frame = cv2.remap(frame, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return frame


def apply_thermal(frame, faces):
    """
    Thermal / heat-vision camera effect.
    Converts to grayscale then applies a JET colour map.
    Faces get highlighted with extra brightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brighten detected face regions to make them "hot"
    for (x, y, fw, fh) in faces:
        face_roi = gray[y:y+fh, x:x+fw]
        gray[y:y+fh, x:x+fw] = cv2.add(face_roi, 40)

    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


def apply_glitch(frame, faces):
    """
    VHS / glitch effect: RGB channel offset, horizontal block shifts,
    and scanline noise. Creates a corrupted digital look.
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # RGB channel offset
    shift = int(5 + 3 * np.sin(time.time() * 8))
    b, g, r = cv2.split(result)
    # Shift red channel right, blue channel left
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    result = cv2.merge([b, g, r])

    # Random horizontal block glitches (3-6 blocks)
    np.random.seed(int(time.time() * 10) % 10000)
    num_glitches = np.random.randint(3, 7)
    for _ in range(num_glitches):
        gy = np.random.randint(0, h - 20)
        gh = np.random.randint(5, 25)
        gshift = np.random.randint(-30, 30)
        if gy + gh < h:
            result[gy:gy+gh] = np.roll(result[gy:gy+gh], gshift, axis=1)

    # Scanlines — darken every other row
    scanlines = np.ones_like(result, dtype=np.float32)
    scanlines[::2] = 0.85
    result = (result.astype(np.float32) * scanlines).astype(np.uint8)

    # Slight green tint for that VHS feel
    result[:, :, 1] = cv2.add(result[:, :, 1], 12)

    return result


def apply_pop_art(frame, faces):
    """
    Andy Warhol–style pop art: posterize the image and tile it
    in a 2×2 grid with different colour tints.
    """
    h, w = frame.shape[:2]

    # Posterize — reduce to limited colours
    div = 64
    poster = (frame // div) * div + div // 2

    # Edge detection for comic outlines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    poster = cv2.bitwise_and(poster, edges_bgr)

    # Create 4 colour-tinted versions
    tints = [
        (0.3, 0.3, 1.5),   # Red tint
        (1.5, 0.3, 1.2),   # Blue-pink tint
        (0.3, 1.5, 1.2),   # Green tint
        (1.2, 1.0, 0.3),   # Cyan tint
    ]

    half_h, half_w = h // 2, w // 2
    small = cv2.resize(poster, (half_w, half_h))
    canvas = np.zeros_like(frame)

    for i, (tb, tg, tr) in enumerate(tints):
        tinted = small.copy().astype(np.float32)
        tinted[:, :, 0] = np.clip(tinted[:, :, 0] * tb, 0, 255)
        tinted[:, :, 1] = np.clip(tinted[:, :, 1] * tg, 0, 255)
        tinted[:, :, 2] = np.clip(tinted[:, :, 2] * tr, 0, 255)
        tinted = tinted.astype(np.uint8)

        row = (i // 2) * half_h
        col = (i % 2) * half_w
        canvas[row:row+half_h, col:col+half_w] = tinted

    return canvas


def apply_mirror(frame, faces):
    """
    Kaleidoscope / mirror effect on detected faces.
    Takes the left half and mirrors it to the right, creating
    a perfectly symmetric face.
    """
    for (x, y, fw, fh) in faces:
        pad = int(fw * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + fw + pad)
        y2 = min(frame.shape[0], y + fh + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        rh, rw = roi.shape[:2]
        half = rw // 2

        if half > 0:
            left_half = roi[:, :half]
            right_mirror = cv2.flip(left_half, 1)

            # Handle odd width
            if right_mirror.shape[1] != rw - half:
                right_mirror = cv2.resize(right_mirror, (rw - half, rh))

            roi[:, half:] = right_mirror
            frame[y1:y2, x1:x2] = roi

    return frame


def apply_emboss(frame, faces):
    """
    3D emboss / relief effect on faces. Gives a carved-in-stone look
    with strong directional lighting.
    """
    for (x, y, fw, fh) in faces:
        pad = int(fw * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + fw + pad)
        y2 = min(frame.shape[0], y + fh + pad)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        embossed = cv2.filter2D(face_roi, -1, kernel)
        embossed = cv2.add(embossed, 128)

        # Blend with a golden tint
        tint = embossed.astype(np.float32)
        tint[:, :, 0] *= 0.7   # reduce blue
        tint[:, :, 2] *= 1.2   # boost red for warmth
        embossed = np.clip(tint, 0, 255).astype(np.uint8)

        frame[y1:y2, x1:x2] = embossed
    return frame


def apply_watercolor(frame, faces):
    """
    Dreamy watercolour painting effect.
    Applies heavy bilateral filtering and edge-preserving smoothing
    for a soft, painted aesthetic across the full frame.
    """
    # Multiple passes of bilateral filter for strong smoothing
    result = frame.copy()
    for _ in range(3):
        result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)

    # Edge-preserving filter for painterly strokes
    result = cv2.edgePreservingFilter(result, flags=1, sigma_s=60, sigma_r=0.4)

    # Slight warm tint
    result = result.astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.05, 0, 255)  # tiny red boost
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.95, 0, 255)  # tiny blue reduction
    result = result.astype(np.uint8)

    return result


# ── Map of AI filter names → apply functions ────────────────────────────────
AI_FILTERS = {
    "pixel_art":  apply_pixel_art,
    "neon_glow":  apply_neon_glow,
    "sketch":     apply_sketch,
    "swirl":      apply_swirl,
    "thermal":    apply_thermal,
    "glitch":     apply_glitch,
    "pop_art":    apply_pop_art,
    "mirror":     apply_mirror,
    "emboss":     apply_emboss,
    "watercolor": apply_watercolor,
}


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_filter, latest_frame
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data"}), 400

    try:
        # Decode base64 image
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── Low-light adaptive enhancement ──
        frame = enhance_low_light(frame)

        # Read the current filter in a thread-safe way
        with filter_lock:
            active_filter = current_filter

        # ── Emoji overlay filters ──
        if active_filter in assets and assets.get(active_filter) is not None:
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

        # ── AI / CV-based creative filters ──
        elif active_filter in AI_FILTERS:
            faces = detect_faces(frame)
            frame = AI_FILTERS[active_filter](frame, faces)

        # Store the latest processed frame for capture
        with frame_lock:
            latest_frame = frame.copy()

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        return jsonify({
            "status": "ok",
            "image": f"data:image/jpeg;base64,{jpg_b64}"
        })
    except Exception as e:
        print("Error processing frame:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set_filter', methods=['POST'])
def set_filter():
    global current_filter
    data = request.get_json()
    new_filter = data.get('filter', 'none')
    with filter_lock:
        current_filter = new_filter
    print(f"Filter changed to: {new_filter}")
    return jsonify({"status": "ok", "filter": new_filter})

@app.route('/capture', methods=['POST'])
def capture():
    """Capture the current processed frame and return it as a JPEG."""
    with frame_lock:
        frame = latest_frame

    if frame is None:
        return jsonify({"status": "error", "message": "No frame available yet"}), 503

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    jpg_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return jsonify({
        "status": "ok",
        "image": f"data:image/jpeg;base64,{jpg_b64}"
    })

@app.route('/post_capture', methods=['POST'])
def post_capture():
    """Save a captured photo to the gallery hub."""
    global post_id_counter
    data = request.get_json()
    image_data = data.get('image', '')
    filter_name = data.get('filter', 'none')
    caption = data.get('caption', '')

    if not image_data:
        return jsonify({"status": "error", "message": "No image data"}), 400

    with gallery_lock:
        post_id_counter += 1
        post = {
            "id": post_id_counter,
            "image": image_data,
            "filter": filter_name,
            "caption": caption,
            "timestamp": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
            "likes": 0,
        }
        gallery_posts.insert(0, post)  # newest first

        # Keep gallery to a reasonable size
        if len(gallery_posts) > 50:
            gallery_posts.pop()

    return jsonify({"status": "ok", "post": post})

@app.route('/gallery_feed')
def gallery_feed():
    """Return all gallery posts as JSON."""
    with gallery_lock:
        return jsonify({"posts": gallery_posts})

@app.route('/like_post', methods=['POST'])
def like_post():
    """Toggle a like on a gallery post."""
    data = request.get_json()
    post_id = data.get('id')
    with gallery_lock:
        for post in gallery_posts:
            if post['id'] == post_id:
                post['likes'] += 1
                return jsonify({"status": "ok", "likes": post['likes']})
    return jsonify({"status": "error", "message": "Post not found"}), 404

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
