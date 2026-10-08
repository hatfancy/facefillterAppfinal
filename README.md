# Real-Time AR Face Filter Web App

A Python-based web application that performs real-time face detection and overlays interactive AR emoji filters. This project uses Flask for the backend server, WebSockets for real-time video streaming, and MediaPipe with OpenCV for accurate face detection.

## Features

- **Real-Time Video Streaming**: Captures webcam stream from the browser, processing frames on the backend.
- **Fast Face Detection**: Uses Google's MediaPipe for highly accurate and fast face tracking.
- **Interactive AR Filters**: Toggle seamlessly between "No Filter", "Smiley", "Wink", and "Cool Guy" emoji filters.
- **Responsive UI**: A modern, dark-themed responsive UI built with pure HTML/CSS/JS.

## Prerequisites

- Python 3.8+
- A working webcam

## Setup Instructions

1. **Install Dependencies**

   It is recommended to use a virtual environment. Install the required Python packages using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate / Download Assets**
   
   If the `assets` folder does not contain the required emoji images (`smiley.png`, `wink.png`, `cool.png`), run the provided download script which fetches open-source emojis (Twemoji):

   ```bash
   python download_assets.py
   ```

3. **Run the Application**

   Start the Flask backend server:

   ```bash
   python app.py
   ```

4. **Test the Application**

   Open your web browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

   - The browser will ask for webcam permission. Please click **Allow**.
   - You will see the video feed. Use the buttons below the video to toggle the face filters.

## Tech Stack Overview

- **Backend**: Python, Flask, Flask-SocketIO
- **Computer Vision**: OpenCV (`cv2`), MediaPipe Face Detection
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript
- **Communication**: WebSockets (Socket.IO) for low-latency frame streaming.
