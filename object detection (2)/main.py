import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
import mediapipe as mp
import math
from tkinter import *
from tkinter import ttk
from ultralytics import YOLO
from tkinter import messagebox
from PIL import Image, ImageTk
import pyttsx3
import speech_recognition as sr
import os

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def object_detection(engine):
    model = YOLO('C:\\Users\\DELL 7390\\OneDrive\\Desktop\\object detection (2)\\object detection (2)\\object detection\\yolow\\yolov8l.pt')
    classNames=open('C:\\Users\\DELL 7390\\OneDrive\\Desktop\\object detection (2)\\object detection (2)\\object detection\\yolow\\objname\\object.name')
    label=classNames.read().split('\n')
    classNames.close()
    print(label)
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        result = model(img, stream=True)
        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                text = f'{label[cls]} {conf}'
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                engine.say(text)
                engine.runAndWait()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        panel.img = img
        panel.config(image=img)
        panel.image = img

def hand_gesture_recognition(engine):
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    model = load_model('C:\\Users\\DELL 7390\\OneDrive\\Desktop\\object detection (2)\\object detection (2)\\object detection\\yolowebcam\\mp_hand_gesture')

    f = open('C:\\Users\\DELL 7390\\OneDrive\\Desktop\\object detection (2)\\object detection (2)\\object detection\\yolowebcam\\gesture.names')
    label = f.read().split('\n')
    f.close()

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        x, y, c = frame.shape

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)

        class_name = ''
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                mp_draw.draw_landmarks(frame, handslms, mp_hand.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])

                classID = np.argmax(prediction)
                class_name = label[classID].capitalize()

        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        engine.say(class_name)
        engine.runAndWait()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)

        panel.img = frame
        panel.config(image=frame)
        panel.image = frame

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_user_input():
    while True:
        choice = choice_var.get()
        if choice:
            if choice == 'object detection':
                object_detection(engine)
            elif choice == 'guessing hand gesture':
                hand_gesture_recognition(engine)
            else:
                messagebox.showerror("Error", "Invalid choice.")
        else:
            messagebox.showerror("Error", "Please select an option.")

def start_detection():
    threading.Thread(target=get_user_input).start()

def main():
    root = Tk()
    root.title("Object Detection & Hand Gesture Recognition")
    root.geometry("1300x650")
    root.iconbitmap(os.path.join(os.path.dirname(__file__), 'C:\\Users\\DELL 7390\\OneDrive\\Desktop\\object detection (2)\\object detection (2)\\object detection\\icon\\Screenshot 2024-04-13 164228.png'))

    style = ttk.Style()
    style.theme_use('clam')

    global choice_var
    choice_var = StringVar(root)

    frame = Frame(root, bg="#90EE90")
    frame.pack(fill=BOTH, expand=True)

    label = Label(frame, text="Select an option:", bg="#90EE90", font=("Helvetica", 18))
    label.pack(pady=(20, 10))

    option_menu = ttk.Combobox(frame, textvariable=choice_var, values=("object detection", "guessing hand gesture"), state="readonly", background="yellow", font=("Helvetica", 16))
    option_menu.pack(pady=(0, 20))

    start_button = Button(frame, text="Start", command=start_detection, bg="black", fg="white", font=("Helvetica", 18))
    start_button.pack(pady=(0, 20), ipadx=10)

    global panel
    panel = Label(frame, bg="#90EE90")
    panel.pack(fill=BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()
