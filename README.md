# Object Detection and Gesture Recognition

Real‑time computer vision demo that streams your webcam to a Flask backend and a React frontend.
- Object detection: Ultralytics YOLOv8. The UI shows only the closest object with a modern overlay and a short info panel.
- Hand gestures: MediaPipe Hands + local heuristics + optional Gemini assistance with caching.

## Features
- YOLOv8 object detection (closest box, neon corners, info panel).
- Hand tracking with MediaPipe.
- Gesture labels from:
  1) Local model (if present),
  2) Lightweight offline heuristics (Thumbs Up/Down, Open Palm, Fist, Peace, OK Hand),
  3) Gemini API for unknown gestures, with single‑flight requests and a persistent cache.
- Gemini key failover and “Fetching gesture…” UX while waiting.

## Requirements
- Python 3.10 (recommended)
- Node 18+ (only if rebuilding the frontend)

Python packages are pinned in `requirment.txt` (typo preserved to match file name).

## Quick start (Windows)
1) Create venv, install deps, and set API key(s):
   - The script installs Python deps and sets `GEMINI_API_KEY` (can be overridden via ENV).

2) Start the backend (Flask):
   - From the root, run your usual start command (e.g., `python "object detection (2)\app.py"`).

3) Open the frontend:
   - The React app is already built under `frontend/dist`. The Flask app serves `/video_feed` as an MJPEG stream that the UI consumes.

## Configuration
- YOLO weight path: `object detection/yolow/yolov8l.pt`
- Gesture label list: `object detection/yolowebcam/gesture.names`
- Optional Gemini keys (used for gesture descriptions/classification):
  - Primary key is read from `GEMINI_API_KEY` env var.
  - Extra keys can be added in code (see `GEMINI_EXTRA_KEYS` in `main.py`). The code tries keys sequentially per request if a key fails.

## How gesture recognition works
1) MediaPipe detects hand landmarks per frame.
2) We try in this order:
   - Local Keras/SavedModel if available.
   - Offline heuristic rules (Thumbs Up/Down, Open Palm, Fist, Peace, OK Hand).
   - Gemini (only when needed):
     - Single‑flight: if a new shape appears, we send one request and display “Fetching gesture…”.
     - Landmark signature cache: the returned label is saved to `object detection/yolowebcam/gesture_cache.json` and reused instantly next time.
     - Left/right normalization: left hands are mirrored before sending to Gemini.

## Object detection overlay
- Only the closest object (largest area) is highlighted.
- Futuristic neon corner box.
- Info panel with short facts, optionally fetched from Gemini.

## Troubleshooting
- No gesture text on screen:
  - Ensure good lighting and that your whole hand is in frame.
  - Try a heuristic gesture first (Thumbs Up/Down, Open Palm, Fist, Peace, OK Hand).
- Gemini quota errors (429):
  - The app tries next API key automatically per request.
  - Cached gestures continue to work offline.
- Missing Python packages in editor warnings: run the venv via `setup_env.bat`.

## Useful files
- Backend: `object detection (2)/app.py`, `object detection (2)/main.py`
- Frontend build: `object detection (2)/frontend/dist`
- Models and labels: `object detection (2)/object detection/yolow*`

## Extending gestures
- Add a line to `gesture.names` for the new label.
- Restart the backend (or let it hot‑reload). The Gemini options auto‑sync with this list.
- For better offline behavior, ask us to add a small heuristic for your new gesture.


