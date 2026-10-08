import cv2
import os

for name in ['smiley', 'wink', 'cool']:
    path = f'assets/{name}.png'
    exists = os.path.exists(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if exists else None
    shape = img.shape if img is not None else None
    print(f"{name}: exists={exists}, shape={shape}")

# Test face detection on a single webcam frame
print("\nTesting webcam + face detection...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Faces detected (minNeighbors=5): {len(faces)}")
        faces2 = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        print(f"Faces detected (minNeighbors=3): {len(faces2)}")
    else:
        print("Failed to read frame")
    cap.release()
else:
    print("Could not open webcam (might be in use by the app)")
