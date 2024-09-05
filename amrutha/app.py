from flask import Flask, render_template, jsonify
import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import threading
import time

app = Flask(__name__)

# Initialize necessary global variables
video_capture = cv2.VideoCapture(0)
recording = False
gaze_detection_thread = None

# Load necessary models for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("open_cv/shape_predictor_68_face_landmarks.dat")  # Ensure the .dat file is in the same folder

# Constants for gaze detection
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5
EYE_AR_THRESH = 0.2
CHECK_INTERVAL = 3
PICTURE_THRESHOLD = 5

# Initialize variables for motion detection
prev_gray = None
face_motions = {}
last_check_time = time.time()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_gaze(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < EYE_AR_THRESH:
        return False, "Eyes Closed"

    left_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES], axis=0)
    right_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES], axis=0)
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)
    
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])
    face_width = landmarks.part(16).x - landmarks.part(0).x
    
    distance = np.linalg.norm(eyes_center - nose_bridge)
    looking_at_camera = (distance / face_width) < GAZE_THRESHOLD
    
    return looking_at_camera, "Looking at camera" if looking_at_camera else "Not looking at camera"

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces

def process_video():
    global recording, prev_gray, last_check_time, face_motions
    gaze_history = deque(maxlen=HISTORY_LENGTH)
    
    while recording:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_cv = detect_bounding_box(frame)
        faces_dlib = detector(gray)

        # Gaze detection and other processing logic goes here (same as your provided script)
        # Skipping the repetitive code for brevity...

        cv2.imshow("Face, Motion, and Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, gaze_detection_thread
    if not recording:
        recording = True
        gaze_detection_thread = threading.Thread(target=process_video)
        gaze_detection_thread.start()
        return jsonify({"message": "Recording started."})
    else:
        return jsonify({"message": "Already recording."})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    if recording:
        recording = False
        return jsonify({"message": "Recording stopped."})
    else:
        return jsonify({"message": "Recording is not active."})

@app.route('/ask_question', methods=['GET'])
def ask_question():
    # Example logic for asking a question
    return jsonify({"message": "Question asked."})

if __name__ == '__main__':
    app.run(debug=True)