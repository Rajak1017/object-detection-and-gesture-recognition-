from flask import Flask, Response, send_from_directory, request, jsonify
from flask_cors import CORS
import cv2
import os
import sys
import json
import time
import queue
import threading
import pyttsx3
import base64 # Import base64 for encoding

# Add the parent directory to the system path to import main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import main

# Define the path to the React build directory
react_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend/dist')) # Changed 'build' to 'dist' based on common Vite output

app = Flask(__name__, static_folder=react_build_dir, template_folder=react_build_dir)
CORS(app)  # Enable CORS for all routes

# Initialize pyttsx3 engine directly in app.py
engine = pyttsx3.init()

# Speech Queue and Worker Thread
speech_queue = queue.Queue()
speech_lock = threading.Lock()

def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Sentinel value to stop the thread
            break
        with speech_lock:
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

class VideoStreamer:
    def __init__(self):
        self.cap = None
        self.current_mode = 'object_detection' # Default mode
        self.last_frame_data = {'frame': '', 'detections': [], 'gesture': ''}
        self.last_frame_lock = threading.Lock()
        self.running = False
        self.thread = None

    def _process_frames(self):
        print(f"VideoStreamer: Attempting to start camera for {self.current_mode} mode...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows compatibility
        if not self.cap.isOpened():
            print(f"Error: Could not open video stream. self.cap: {self.cap} (Type: {type(self.cap)})\n"+
                  "Please ensure your webcam is connected and not in use by another application.")
            self.running = False
            return
        else:
            print(f"VideoStreamer: Camera successfully opened for mode {self.current_mode}. Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        if self.current_mode == 'object_detection':
            for frame_bytes, detections in main.object_detection(self.cap, engine):
                if not self.running:
                    break
                for det in detections:
                    if 'speech_text' in det:
                        speech_queue.put(det['speech_text'])
                with self.last_frame_lock:
                    self.last_frame_data = {
                        'frame': base64.b64encode(frame_bytes).decode('utf-8'),
                        'detections': detections,
                        'gesture': ''
                    }
        elif self.current_mode == 'hand_gesture_recognition':
            for frame_bytes, gesture_name in main.hand_gesture_recognition(self.cap, engine):
                if not self.running:
                    break
                if gesture_name:
                    speech_queue.put(gesture_name)
                if frame_bytes:
                    with self.last_frame_lock:
                        self.last_frame_data = {
                            'frame': base64.b64encode(frame_bytes).decode('utf-8'),
                            'detections': [],
                            'gesture': gesture_name
                        }
                else:
                    print("VideoStreamer: Received empty frame_bytes in hand_gesture_recognition mode.")
        
        if self.cap:
            self.cap.release()
        print("VideoStreamer: Stopped.")

    def start(self, mode):
        if not self.running:
            self.current_mode = mode
            self.running = True
            self.thread = threading.Thread(target=self._process_frames, daemon=True)
            self.thread.start()
            print(f"VideoStreamer: Thread started for {mode}.")
            return True
        else:
            # If already running, and mode is different, restart it.
            if self.current_mode != mode:
                self.stop()
                self.current_mode = mode
                self.running = True
                self.thread = threading.Thread(target=self._process_frames, daemon=True)
                self.thread.start()
                print(f"VideoStreamer: Restarted thread for {mode}.")
                return True
            print(f"VideoStreamer: Already running in {self.current_mode} mode.")
            return False # Already running in the requested mode

    def stop(self):
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5) # Wait for the thread to finish
                if self.thread.is_alive():
                    print("Warning: VideoStreamer thread did not terminate gracefully.")
            if self.cap:
                self.cap.release()
                self.cap = None
                time.sleep(1)  # Wait a moment to ensure camera is released
            self.last_frame_data = {'frame': '', 'detections': [], 'gesture': ''}
            print("VideoStreamer: Thread stopped.")
            return True
        return False

    def get_latest_frame_data(self):
        with self.last_frame_lock:
            return self.last_frame_data

videostreamer = VideoStreamer()

@app.route('/')
def index():
    return send_from_directory(react_build_dir, 'index.html')

# Serve static files for the React app
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(react_build_dir, filename)

@app.route('/video_feed')
def video_feed():
    return jsonify(videostreamer.get_latest_frame_data())

@app.route('/start_video_processing', methods=['POST'])
def start_video_processing():
    mode = request.json.get('mode', 'object_detection')
    if videostreamer.start(mode):
        return jsonify({'status': 'started', 'mode': mode}), 200
    else:
        return jsonify({'status': 'already_running', 'mode': videostreamer.current_mode}), 200

@app.route('/stop_video_processing', methods=['POST'])
def stop_video_processing():
    if videostreamer.stop():
        return jsonify({'status': 'stopped'}), 200
    else:
        return jsonify({'status': 'not_running'}), 200

if __name__ == '__main__':
    # Before running the app, ensure the speech worker can be gracefully shut down
    import atexit
    atexit.register(lambda: speech_queue.put(None)) # Put sentinel on exit
    atexit.register(lambda: videostreamer.stop()) # Ensure videostreamer stops on exit
    app.run(host='0.0.0.0', port=5000, debug=True)
