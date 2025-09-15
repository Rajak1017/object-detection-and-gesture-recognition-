import cv2
import numpy as np
import threading
import tensorflow as tf # Import tensorflow
from tensorflow.keras.models import load_model
import mediapipe as mp
import math
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import os
import time
from functools import lru_cache
try:
    import google.generativeai as genai
except Exception as import_error:
    print(f"Gemini: google-generativeai import failed: {import_error}")
    genai = None

# Initialize pyttsx3 engine

GEMINI_DEFAULT_API_KEY = "ADD_YOUR_GEMINI_API_KEY_HERE"
# Additional user-provided keys for rotation
GEMINI_EXTRA_KEYS = [
    "ADD_YOUR_GEMINI_API_KEY_HERE",
    "ADD_YOUR_GEMINI_API_KEY_HERE",
]


_gemini_key_cooldowns = {}
_gemini_current_key = None


def _pick_active_key():
    # Priority: env var -> default -> extras, skipping cooled-down keys
    candidates = []
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        candidates.append(env_key)
    if GEMINI_DEFAULT_API_KEY:
        candidates.append(GEMINI_DEFAULT_API_KEY)
    candidates.extend(GEMINI_EXTRA_KEYS)

    now = time.time()
    for key in candidates:
        until = _gemini_key_cooldowns.get(key, 0)
        if now >= until and key:
            return key
    return None


def _configure_gemini_model():
    if genai is None:
        print("Gemini: SDK unavailable; skipping online info.")
        return None
    api_key = _pick_active_key()
    if not api_key:
        print("Gemini: No API key provided.")
        return None
    try:
        genai.configure(api_key=api_key)
        global _gemini_current_key
        _gemini_current_key = api_key
        # Try preferred model first, then fall back to a widely available one
        try:
            print("Gemini: Trying model 'gemini-2.0-flash'.")
            return genai.GenerativeModel("gemini-2.0-flash")
        except Exception as m1_err:
            print(f"Gemini: 'gemini-2.0-flash' unavailable: {m1_err}; trying 'gemini-1.5-flash'.")
            return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as cfg_err:
        print(f"Gemini: Configuration failed: {cfg_err}")
        return None


_gemini_model = _configure_gemini_model()
_gemini_cooldown_until_ts = 0.0  # epoch seconds; skip Gemini calls while in cooldown

# Single-flight lock and cache for gesture results
_inflight_gesture_key = None
_inflight_started_ts = 0.0
_gesture_cache = {}
_gesture_cache_path = os.path.join(os.path.dirname(__file__), 'object detection/yolowebcam/gesture_cache.json')

def _load_gesture_cache():
    global _gesture_cache
    try:
        import json
        if os.path.exists(_gesture_cache_path):
            with open(_gesture_cache_path, 'r', encoding='utf-8') as jf:
                _gesture_cache = json.load(jf)
    except Exception:
        _gesture_cache = {}

def _save_gesture_cache():
    try:
        import json
        with open(_gesture_cache_path, 'w', encoding='utf-8') as jf:
            json.dump(_gesture_cache, jf)
    except Exception:
        pass

_load_gesture_cache()


@lru_cache(maxsize=256)
def _fetch_object_info_cached(class_name: str) -> str:
    if not _gemini_model:
        return "No online info available."
    prompt = f"Give a concise 1-2 sentence description of a '{class_name}'. No markdown, no lists."
    try:
        resp = _gemini_model.generate_content(prompt)
        # Try standard text field first
        text = (getattr(resp, 'text', None) or "").strip()
        # Fallback: extract from candidates/parts structure if present
        if not text and hasattr(resp, 'candidates') and resp.candidates:
            for cand in resp.candidates:
                content = getattr(cand, 'content', None)
                parts = getattr(content, 'parts', None) if content else None
                if parts:
                    for part in parts:
                        part_text = getattr(part, 'text', None)
                        if part_text and part_text.strip():
                            text = part_text.strip()
                            break
                if text:
                    break
        if not text:
            return "No info available."
        return text[:400]
    except Exception as gen_err:
        print(f"Gemini: generate_content failed: {gen_err}")
        return "No info available."


def object_detection(cap, engine):
    model = YOLO(os.path.join(os.path.dirname(__file__), 'object detection/yolow/yolov8l.pt'))
    classNames=open(os.path.join(os.path.dirname(__file__), 'object detection/yolow/objname/object.name'))
    label=classNames.read().split('\n')
    classNames.close()
    print(label)
    # cap = cv2.VideoCapture(0) # Removed: camera now passed as argument
    last_class_name = None
    last_info_text = ""
    last_fetch_ts = 0.0
    fetch_interval_sec = 15.0

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("object_detection: Failed to read frame from camera.")
            break
        result = model(img, stream=True)
        all_detections = []
        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width, height = x2 - x1, y2 - y1
                area = max(width, 0) * max(height, 0)
                conf = float(math.ceil((box.conf[0] * 100)) / 100)
                cls = int(box.cls[0])
                class_name = label[cls] if 0 <= cls < len(label) else str(cls)
                all_detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'box': [x1, y1, x2, y2],
                    'area': area
                })

        # Select only the closest (largest area) detection
        detections = []
        selected = None
        if all_detections:
            selected = max(all_detections, key=lambda d: d['area'])
            detections = [selected]

            # Draw the selected detection
            x1, y1, x2, y2 = selected['box']
            class_name = selected['class']
            conf = selected['confidence']
            def _draw_neon_box(frame, p1, p2, color=(255, 0, 255)):
                x1b, y1b = p1
                x2b, y2b = p2
                # Outer glow using overlay
                overlay_nb = frame.copy()
                cv2.rectangle(overlay_nb, (x1b, y1b), (x2b, y2b), color, 2)
                cv2.addWeighted(overlay_nb, 0.4, frame, 0.6, 0, dst=frame)
                # Corner lines for a futuristic look
                line_len = max(15, int(0.08 * min(x2b - x1b, y2b - y1b)))
                thickness = 3
                # Top-left
                cv2.line(frame, (x1b, y1b), (x1b + line_len, y1b), color, thickness)
                cv2.line(frame, (x1b, y1b), (x1b, y1b + line_len), color, thickness)
                # Top-right
                cv2.line(frame, (x2b, y1b), (x2b - line_len, y1b), color, thickness)
                cv2.line(frame, (x2b, y1b), (x2b, y1b + line_len), color, thickness)
                # Bottom-left
                cv2.line(frame, (x1b, y2b), (x1b + line_len, y2b), color, thickness)
                cv2.line(frame, (x1b, y2b), (x1b, y2b - line_len), color, thickness)
                # Bottom-right
                cv2.line(frame, (x2b, y2b), (x2b - line_len, y2b), color, thickness)
                cv2.line(frame, (x2b, y2b), (x2b, y2b - line_len), color, thickness)

            _draw_neon_box(img, (x1, y1), (x2, y2), color=(255, 0, 255))

            # Fetch live info from Gemini with caching and rate limit
            now = time.time()
            should_fetch = (class_name != last_class_name) or (now - last_fetch_ts >= fetch_interval_sec)
            if should_fetch:
                info_text_try = _fetch_object_info_cached(class_name.lower())
                if info_text_try:
                    last_info_text = info_text_try
                last_fetch_ts = now
                last_class_name = class_name
            info_text = last_info_text or 'No additional info available for this object.'

            # Word-wrapped info panel positioned near box but within frame
            img_h, img_w = img.shape[:2]
            base_panel_w = int(0.45 * img_w)
            max_panel_w = max(280, min(base_panel_w, img_w - 20))
            margin = 12
            header = f"{class_name.capitalize()} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            header_scale = 0.7
            text_scale = 0.55
            header_th = 2
            text_th = 1

            # Wrap info text to fit width
            def _wrap_text(text, max_width):
                words = text.split()
                lines = []
                current = ''
                for w in words:
                    test = (current + ' ' + w).strip()
                    (tw, th), _ = cv2.getTextSize(test, font, text_scale, text_th)
                    if tw <= (max_width - 2 * margin) or not current:
                        current = test
                    else:
                        lines.append(current)
                        current = w
                if current:
                    lines.append(current)
                return lines

            wrapped_lines = _wrap_text(info_text, max_panel_w)
            # Compute panel height
            (_, header_h), _ = cv2.getTextSize(header, font, header_scale, header_th)
            line_heights = [cv2.getTextSize(line, font, text_scale, text_th)[0][1] for line in wrapped_lines]
            lines_total_h = sum(line_heights) + (len(wrapped_lines) * 6)
            panel_h = header_h + lines_total_h + margin * 2 + 8

            # Prefer panel above the box; if not enough space, place below; else top-left
            pref_x = max(10, min(x1, img_w - max_panel_w - 10))
            if y1 - panel_h - 10 >= 0:
                pref_y = y1 - panel_h - 10
            elif y2 + panel_h + 10 <= img_h:
                pref_y = y2 + 10
            else:
                pref_y = 10

            panel_x, panel_y = pref_x, pref_y
            panel_w = max_panel_w

            # Draw translucent panel with neon border
            overlay = img.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
            cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 0, 255), 2)

            # Render header and wrapped text inside
            text_x = panel_x + margin
            text_y = panel_y + margin + header_h
            cv2.putText(img, header, (text_x, text_y), font, header_scale, (255, 255, 255), header_th)
            cursor_y = text_y + 8
            for idx, line in enumerate(wrapped_lines):
                (lw, lh), _ = cv2.getTextSize(line, font, text_scale, text_th)
                cursor_y += lh + 6
                if cursor_y + margin >= panel_y + panel_h:
                    break
                cv2.putText(img, line, (text_x, cursor_y), font, text_scale, (220, 220, 220), text_th)

        # Encode the frame as JPEG
        if img is not None:
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                print("object_detection: Failed to encode frame.")
                continue
            frame_bytes = buffer.tobytes()
            yield frame_bytes, detections
        else:
            print("object_detection: Image is None, skipping frame.")
            continue

    # cap.release() # Removed: camera now managed externally

def hand_gesture_recognition(cap, engine):
    print("hand_gesture_recognition: Function called.")
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_draw = mp.solutions.drawing_utils

    model_path = os.path.join(os.path.dirname(__file__), 'object detection/yolowebcam/mp_hand_gesture')
    
    print(f"hand_gesture_recognition: Attempting to load model from: {model_path}")
    # Prefer loading as a Keras model without compiling to avoid optimizer slot issues
    loaded_model = None
    infer = None
    model_kind = 'unknown'
    try:
        loaded_model = load_model(model_path, compile=False)
        model_kind = 'keras'
        print("hand_gesture_recognition: Keras model loaded successfully (compile=False).")
    except Exception as keras_load_error:
        print(f"hand_gesture_recognition: Keras load failed, will try SavedModel. Error: {keras_load_error}")
        try:
            loaded_model = tf.saved_model.load(model_path)
            infer = loaded_model.signatures.get('serving_default')
            model_kind = 'saved_model'
            print("hand_gesture_recognition: TensorFlow SavedModel loaded successfully.")
        except Exception as saved_model_error:
            print(f"hand_gesture_recognition: Failed to load model. SavedModel error: {saved_model_error}")
            # Proceed without a model; classifications will be empty
            loaded_model = None
            infer = None
            model_kind = 'none'


    f = open(os.path.join(os.path.dirname(__file__), 'object detection/yolowebcam/gesture.names'))
    label = f.read().split('\n')
    f.close()

    # Gemini-based gesture fallback configuration
    # Start with a base set, then merge with gesture.names for auto-sync
    base_gestures = [
        # Core controls
        'thumbs up', 'thumbs down', 'ok hand', 'okay', 'ok', 'open palm', 'stop', 'open hand',
        'peace', 'victory', 'two fingers', 'index up', 'pointing', 'fist', 'rock on', 'rock', 'call me',
        # From image list
        'applause', 'hi hello', 'love you', 'joy', 'surprise', 'gratitude', 'heart hands', 'hi stop',
        'writing hand', 'handshake', 'praying hands', 'strength', 'hug', 'receiving', 'picking up',
        'raised hand', 'good luck', 'greet fight', 'greet support', 'wish to prosper', 'you',
        # Existing extras
        'pinch', 'punch', 'smile', 'clapping hands', 'live long'
    ]
    # Read gesture.names to include custom items
    try:
        with open(os.path.join(os.path.dirname(__file__), 'object detection/yolowebcam/gesture.names'), 'r') as gf:
            file_gestures = [ln.strip().lower() for ln in gf.readlines() if ln.strip()]
    except Exception:
        file_gestures = []
    # Merge and de-duplicate
    gesture_options = sorted(list({*base_gestures, *file_gestures}))
    last_gem_fetch_ts = 0.0
    gem_fetch_interval_sec = 1.0

    def _landmark_signature(points: list) -> str:
        # Normalize landmarks to be translation/scale invariant; return a short signature string
        if not points:
            return ''
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        minx, miny = float(xs.min()), float(ys.min())
        xs -= minx
        ys -= miny
        maxdim = max(float(xs.max()) + 1e-6, float(ys.max()) + 1e-6)
        xs /= maxdim
        ys /= maxdim
        # Downsample to a few key points to keep signature compact
        idxs = np.linspace(0, len(xs) - 1, num=min(14, len(xs)), dtype=int)
        sig = np.concatenate([xs[idxs], ys[idxs]]).round(2)
        return '|' + ','.join(map(lambda v: f"{v:.2f}", sig.tolist()))

    def _classify_simple_gesture(landmarks_px: list, img_height: int) -> str:
        # Simple offline heuristic: open palm, fist, thumbs up/down
        # landmarks_px: list of [x, y] for 21 points (if available)
        if len(landmarks_px) < 21:
            return ''
        # Indices per MediaPipe Hands
        WRIST = 0
        THUMB_TIP, THUMB_IP = 4, 3
        INDEX_TIP, INDEX_PIP = 8, 6
        MIDDLE_TIP, MIDDLE_PIP = 12, 10
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18
        # y grows downwards; smaller y means higher in image
        def extended(tip, pip):
            return landmarks_px[tip][1] < landmarks_px[pip][1] - 5
        def folded(tip, pip):
            return landmarks_px[tip][1] > landmarks_px[pip][1] + 5
        def dist(a, b):
            dx = landmarks_px[a][0] - landmarks_px[b][0]
            dy = landmarks_px[a][1] - landmarks_px[b][1]
            return (dx*dx + dy*dy) ** 0.5
        thumb_ext = extended(THUMB_TIP, THUMB_IP)
        index_ext = extended(INDEX_TIP, INDEX_PIP)
        middle_ext = extended(MIDDLE_TIP, MIDDLE_PIP)
        ring_ext = extended(RING_TIP, RING_PIP)
        pinky_ext = extended(PINKY_TIP, PINKY_PIP)
        num_ext = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])
        if num_ext >= 4:
            return 'Open Palm'
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and (thumb_ext or folded(THUMB_TIP, THUMB_IP)):
            # Determine up vs down by average of finger knuckles
            base_y = sum([landmarks_px[i][1] for i in [5, 9, 13, 17]]) / 4.0
            thumb_y = landmarks_px[THUMB_TIP][1]
            return 'Thumbs Up' if thumb_y < base_y else 'Thumbs Down'
        if num_ext == 0:
            return 'Fist'
        # Peace: index and middle extended, ring and pinky folded
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 'Peace'
        # OK hand: index and thumb touching; other fingers extended
        # Use relative threshold based on wrist-middle base distance for scale
        scale = max(dist(WRIST, 9), 1.0)
        if dist(THUMB_TIP, INDEX_TIP) / scale < 0.35 and middle_ext and ring_ext and pinky_ext:
            return 'OK Hand'
        return ''

    def _gemini_classify_gesture(roi_bgr: np.ndarray, handedness_hint: str = '', cache_key: str = '') -> str:
        if _gemini_model is None:
            return ''
        # Global cooldown after rate-limit errors
        global _gemini_cooldown_until_ts
        now = time.time()
        if now < _gemini_cooldown_until_ts:
            return ''
        # Cache check
        if cache_key and cache_key in _gesture_cache:
            return _gesture_cache[cache_key]
        try:
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode('.jpg', roi_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            bytes_img = buf.tobytes()
            prompt = (
                "You are a gesture classifier. Treat left and right hands as the same gesture. "
                f"If it helps, hand: {handedness_hint or 'unknown'}. "
                f"Return exactly one of these labels: {', '.join(gesture_options)}. "
                "If unsure, return 'Unknown'. No extra text."
            )
            resp = _gemini_model.generate_content([
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": bytes_img}}
            ])
            text = (getattr(resp, 'text', None) or '').strip().lower()
            text = text.replace('.', '').strip()
            for opt in gesture_options:
                if opt in text:
                    if cache_key:
                        _gesture_cache[cache_key] = opt.title()
                        _save_gesture_cache()
                    return opt.title()
            if text in [o.lower() for o in gesture_options]:
                if cache_key:
                    _gesture_cache[cache_key] = text.title()
                    _save_gesture_cache()
                return text.title()
            return 'Unknown'
        except Exception as e:
            # Apply cooldown on quota/rate-limit to avoid spamming the API
            msg = str(e)
            if '429' in msg or 'quota' in msg.lower() or 'rate' in msg.lower():
                # Cool down current key and try rotating to the next
                _gemini_cooldown_until_ts = time.time() + 20 * 60  # 20 minutes global UI cooldown
                try:
                    global _gemini_current_key
                    if _gemini_current_key:
                        _gemini_key_cooldowns[_gemini_current_key] = time.time() + 60 * 60  # 1 hour key cooldown
                    # Attempt reconfigure with next available key
                    new_key = _pick_active_key()
                    if new_key and new_key != _gemini_current_key:
                        genai.configure(api_key=new_key)
                        _gemini_current_key = new_key
                        print("Gemini: Rotated API key due to quota limits.")
                except Exception as rotate_err:
                    print(f"Gemini: Key rotation failed: {rotate_err}")
            print(f"Gemini gesture classify failed: {e}")
            return ''

    # Persist last recognized gesture for display smoothing
    last_display_gesture = ''
    last_display_source = ''
    last_display_ts = 0.0
    display_hold_sec = 1.5

    while True:
        print("hand_gesture_recognition: Inside loop, attempting to read frame.")
        ret, frame = cap.read()
        if not ret or frame is None:
            print("hand_gesture_recognition: Failed to retrieve frame from camera or frame is None.")
            continue
        print("hand_gesture_recognition: Frame successfully retrieved.")
        h, w, c = frame.shape

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)

        class_name = ''
        class_source = ''  # 'Local' or 'Gemini'
        if result.multi_hand_landmarks:
            landmarks = []
            handedness = 'unknown'
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * w)
                    lmy = int(lm.y * h)
                    landmarks.append([lmx, lmy])
                mp_draw.draw_landmarks(frame, handslms, mp_hand.HAND_CONNECTIONS)
            # Try to infer handedness hint from landmark direction (simple heuristic)
            try:
                xs = [pt[0] for pt in landmarks]
                if np.mean(xs[:5]) < np.mean(xs[-5:]):
                    handedness = 'right'
                else:
                    handedness = 'left'
            except Exception:
                handedness = 'unknown'

            # Run inference depending on how the model was loaded
            try:
                if model_kind == 'keras' and loaded_model is not None:
                    # Flatten landmarks to shape (1, 42) for a typical dense network
                    input_array = np.array(landmarks, dtype=np.float32).flatten()[None, :]
                    prediction = loaded_model.predict(input_array, verbose=0)
                    classID = int(np.argmax(prediction, axis=-1))
                    class_name = label[classID].capitalize()
                elif model_kind == 'saved_model' and infer is not None:
                    # Original SavedModel path expects a named tensor
                    input_tensor = tf.constant([landmarks], dtype=tf.float32)
                    predictions_output = infer(flatten_2_input=input_tensor)
                    # Prefer a known key, otherwise take the first tensor
                    if 'dense_16' in predictions_output:
                        prediction = predictions_output['dense_16'].numpy()
                    else:
                        first_tensor = next(iter(predictions_output.values()))
                        prediction = first_tensor.numpy()
                    classID = int(np.argmax(prediction, axis=-1))
                    class_name = label[classID].capitalize()
                if class_name:
                    class_source = 'Local'
            except Exception as inference_error:
                print(f"hand_gesture_recognition: Inference failed: {inference_error}")

            # Local heuristic fallback before calling Gemini
            if not class_name:
                simple_guess = _classify_simple_gesture(landmarks, h)
                if simple_guess:
                    class_name = simple_guess
                    class_source = 'Heuristic'

            # Build cache key from normalized landmarks
            cache_key = _landmark_signature(landmarks)

            # Gemini single-flight: if a request is in-flight for this key, show fetching and skip new calls
            global _inflight_gesture_key, _inflight_started_ts
            is_same_request = (_inflight_gesture_key == cache_key)
            in_flight = is_same_request and ((time.time() - _inflight_started_ts) <= 8.0)

            # Gemini classification: call only if not in-flight and not cached
            if _gemini_model is not None and cache_key and not in_flight and cache_key not in _gesture_cache:
                xs = [pt[0] for pt in landmarks]
                ys = [pt[1] for pt in landmarks]
                min_x, max_x = max(0, min(xs)), min(w - 1, max(xs))
                min_y, max_y = max(0, min(ys)), min(h - 1, max(ys))
                pad_x = int(0.2 * (max_x - min_x + 1))
                pad_y = int(0.2 * (max_y - min_y + 1))
                cx1 = max(0, min_x - pad_x)
                cy1 = max(0, min_y - pad_y)
                cx2 = min(frame.shape[1] - 1, max_x + pad_x)
                cy2 = min(frame.shape[0] - 1, max_y + pad_y)
                roi = frame[cy1:cy2, cx1:cx2]
                # Normalize left-hand ROI by mirroring so Gemini sees consistent orientation
                if handedness == 'left' and roi.size > 0:
                    roi = cv2.flip(roi, 1)
                now_ts = time.time()
                if roi.size > 0 and (now_ts - last_gem_fetch_ts) >= gem_fetch_interval_sec:
                    _inflight_gesture_key = cache_key
                    _inflight_started_ts = now_ts
                    guessed = _gemini_classify_gesture(roi, handedness_hint=handedness, cache_key=cache_key)
                    _inflight_gesture_key = None
                    if guessed:
                        if guessed.lower() != 'unknown' or not class_name:
                            class_name = guessed
                            class_source = 'Gemini'
                            print(f"Gesture (Gemini): {class_name}")
                    last_gem_fetch_ts = now_ts

            # If cached result exists, use it immediately
            if (not class_name) and cache_key in _gesture_cache:
                class_name = _gesture_cache[cache_key]
                class_source = 'Cache'
        else:
            # No hand landmarks; clear in-flight key quickly
            _inflight_gesture_key = None

        # Decide what to display: hold last known gesture briefly if current is empty/unknown
        now_disp = time.time()
        if class_name and class_name.lower() != 'unknown':
            last_display_gesture = class_name
            last_display_source = class_source
            last_display_ts = now_disp
        elif (now_disp - last_display_ts) <= display_hold_sec and last_display_gesture:
            class_name = last_display_gesture
            class_source = last_display_source

        # On-screen gesture banner (top center) with optional action line (no source shown)
        has_gesture = bool(class_name) or (_inflight_gesture_key is not None)
        banner_text_main = (class_name or 'Fetching gesture…') if has_gesture else 'No Gesture'

        # Map gestures to action intents
        intent_map = {
            'thumbs up': "Thumbs up is used to agree or say OK/Yes.",
            'okay': "'OK' sign shows confirmation that everything is fine.",
            'ok': "'OK' sign shows confirmation that everything is fine.",
            'thumbs down': "Thumbs down is used to disagree or say Not OK.",
            'open palm': "Open palm is used to say Stop or a friendly Hello.",
            'open hand': "Open hand is used to say Stop or a friendly Hello.",
            'stop': "Stop gesture asks to pause or halt an action.",
            'peace': "Peace sign expresses peace or the number two.",
            'victory': "Victory sign expresses peace or the number two.",
            'two fingers': "Two fingers often means peace or the number two.",
            'fist': "A closed fist means Ready/Confirm or shows strength.",
            'pointing': "Pointing is used to select or indicate a target.",
            'index up': "Index up asks for attention or means number one.",
            'rock': "Rock gesture expresses excitement — ‘rock on’.",
            'call me': "Call me gesture asks to contact or call the person.",
            'pinch': "Pinch is used for zooming or precise selection.",
            'live long': "Vulcan salute: ‘Live long and prosper’.",
            'punch': "Punch indicates a strike or strong action.",
            'smile': "Smile communicates friendliness or approval.",
            'clapping hands': "Clapping shows applause and appreciation."
        }
        # Provide a friendly default intent if not explicitly mapped
        gkey = banner_text_main.split('  |  ')[0].lower()
        intent_text = ''
        if has_gesture:
            intent_text = intent_map.get(gkey, '')
            if not intent_text:
                # Heuristics for default intent
                if 'thumb' in gkey and 'up' in gkey:
                    intent_text = 'Agree / OK'
                elif 'thumb' in gkey and 'down' in gkey:
                    intent_text = 'Disagree / Not OK'
                elif 'peace' in gkey or 'victory' in gkey or 'two' in gkey:
                    intent_text = 'Peace / Two'
                elif 'stop' in gkey or 'open' in gkey or 'palm' in gkey:
                    intent_text = 'Stop / Hello'
                elif 'point' in gkey or 'index' in gkey:
                    intent_text = 'Select / Point'
                elif 'fist' in gkey or 'punch' in gkey:
                    intent_text = 'Confirm / Ready'
                elif 'call' in gkey:
                    intent_text = 'Call Me'
                elif 'clap' in gkey:
                    intent_text = 'Clap / Applause'
                elif 'smile' in gkey:
                    intent_text = 'Smile / Friendly'
                else:
                    intent_text = 'Gesture Detected'

        font_main = cv2.FONT_HERSHEY_SIMPLEX
        font_sub = cv2.FONT_HERSHEY_SIMPLEX

        # Responsive max width (80% of frame, with hard min/max)
        frame_w = frame.shape[1]
        max_pill_w = max(260, min(int(0.8 * frame_w), frame_w - 20))

        def wrap_text_lines(text, font, scale, thickness, max_width):
            if not text:
                return [], 0
            words = text.split()
            lines = []
            current = ''
            max_line_w = 0
            for wtok in words:
                trial = (current + ' ' + wtok).strip()
                (tw, th), _ = cv2.getTextSize(trial, font, scale, thickness)
                if tw <= max_width or not current:
                    current = trial
                    max_line_w = max(max_line_w, tw)
                else:
                    if current:
                        lines.append(current)
                    current = wtok
                    (tw2, _), _ = cv2.getTextSize(current, font, scale, thickness)
                    max_line_w = max(max_line_w, tw2)
            if current:
                lines.append(current)
            return lines, max_line_w

        # Compute wrapped lines for main and intent
        main_lines, main_w = wrap_text_lines(banner_text_main, font_main, 0.9, 2, max_pill_w - 60)
        sub_lines, sub_w = wrap_text_lines(intent_text, font_sub, 0.7, 2, max_pill_w - 60) if intent_text else ([], 0)

        # Measure heights
        line_gap = 6
        line_heights_main = [cv2.getTextSize(l, font_main, 0.9, 2)[0][1] for l in main_lines] or [0]
        line_heights_sub = [cv2.getTextSize(l, font_sub, 0.7, 2)[0][1] for l in sub_lines]
        main_h = sum(line_heights_main) + (len(main_lines) - 1) * line_gap
        sub_h_total = (sum(line_heights_sub) + (len(sub_lines) - 1) * line_gap) if sub_lines else 0
        text_block_w = max(main_w, sub_w)
        total_h = main_h + (sub_h_total + 8 if sub_lines else 0)

        # Rounded pill container with soft shadow and optional icon
        pad_x, pad_y = 16, 14
        pill_w = min(max_pill_w, text_block_w + pad_x * 2 + 30)  # +30 for icon and spacing
        pill_h = total_h + pad_y * 2
        cx = frame.shape[1] // 2
        top_y = 36
        pill_x1 = max(10, cx - pill_w // 2)
        pill_y1 = top_y
        pill_x2 = min(frame.shape[1] - 10, pill_x1 + pill_w)
        pill_y2 = pill_y1 + pill_h

        def draw_rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
            x1, y1 = pt1
            x2, y2 = pt2
            if thickness < 0:
                # filled
                cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
                cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
                cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
                cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
                cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
                cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
            else:
                cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
                cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
                cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
                cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
                cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
                cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
                cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
                cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        # Shadow
        overlay_shadow = frame.copy()
        draw_rounded_rect(overlay_shadow, (pill_x1 + 3, pill_y1 + 4), (pill_x2 + 3, pill_y2 + 4), 14, (0, 0, 0), -1)
        cv2.addWeighted(overlay_shadow, 0.25, frame, 0.75, 0, dst=frame)
        # Pill background
        overlay_pill = frame.copy()
        bg_color = (30, 30, 30)
        draw_rounded_rect(overlay_pill, (pill_x1, pill_y1), (pill_x2, pill_y2), 16, bg_color, -1)
        cv2.addWeighted(overlay_pill, 0.70, frame, 0.30, 0, dst=frame)
        # Border glow
        border_color = (90, 240, 180) if has_gesture else (0, 210, 255)
        draw_rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), 16, border_color, 2)

        # Optional icon: small circle at left colored by state
        icon_r = 8
        icon_cx = pill_x1 + 18
        icon_cy = pill_y1 + pill_h // 2
        icon_color = (90, 240, 180) if has_gesture else (0, 210, 255)
        cv2.circle(frame, (icon_cx, icon_cy), icon_r, icon_color, -1)

        # Text block inside pill
        text_x = icon_cx + 18
        cursor_y = pill_y1 + pad_y
        # Draw main (possibly multi-line)
        for idx, line in enumerate(main_lines):
            (_, lh), _ = cv2.getTextSize(line, font_main, 0.9, 2)
            cursor_y += lh
            if cursor_y + pad_y > pill_y2:
                break
            cv2.putText(frame, line, (text_x, cursor_y), font_main, 0.9, (255, 255, 255), 2)
            cursor_y += line_gap
        # Draw sub lines
        if sub_lines:
            cursor_y += 2
            for line in sub_lines:
                (_, lh2), _ = cv2.getTextSize(line, font_sub, 0.7, 2)
                cursor_y += lh2
                if cursor_y + pad_y > pill_y2:
                    break
                cv2.putText(frame, line, (text_x, cursor_y), font_sub, 0.7, (180, 255, 255), 2)
                cursor_y += line_gap

        # Log intent when gesture changes and intent exists
        if class_name and class_name.lower() != 'unknown' and intent_text and (class_name == last_display_gesture) and (now_disp == last_display_ts or (now_disp - last_display_ts) <= display_hold_sec):
            print(f"Gesture intent: {intent_text}")

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret or buffer is None:
            print("hand_gesture_recognition: Failed to encode frame to JPEG.")
            continue
        frame_bytes = buffer.tobytes()
        # Send the stabilized gesture to the frontend so it matches the on-screen banner
        output_gesture = class_name or last_display_gesture or ''
        yield frame_bytes, output_gesture

if __name__ == "__main__":
    print("Backend logic ready. No GUI will be displayed.")
    print("To use object detection, call object_detection(engine) in a streaming context.")
    print("To use hand gesture recognition, call hand_gesture_recognition(engine) in a streaming context.")
