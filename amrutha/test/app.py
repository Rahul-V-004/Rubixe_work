import random
import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import time
import threading
from flask import Flask, Response, render_template, jsonify, url_for
import sounddevice as sd
import soundfile as sf
import wave
import queue

app = Flask(__name__)

# Video processing globals
video_capture = None
output_frame = None
video_lock = threading.Lock()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

# Audio processing globals
audio_queue = queue.Queue()
audio_stream = None
fs = 44100  # Sample rate

# Video processing functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_center(landmarks, eye_indices):
    return np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices], axis=0)

def detect_gaze(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EYE_AR_THRESH:
        return False, "Eyes Closed"
    
    left_eye_center = get_eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE_INDICES)
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

def process_frame(frame):
    global prev_gray, last_check_time, face_motions

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_cv = detect_bounding_box(frame)
    faces_dlib = detector(gray)

    face_areas = [w * h for (x, y, w, h) in faces_cv]
    main_interviewer_idx = face_areas.index(max(face_areas)) if face_areas else None

    current_time = time.time()

    if current_time - last_check_time >= CHECK_INTERVAL:
        last_check_time = current_time

        if prev_gray is not None and len(faces_cv) > 0:
            for i, (x, y, w, h) in enumerate(faces_cv):
                face_id = f"face_{i}"

                face_diff = cv2.absdiff(prev_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                _, face_thresh = cv2.threshold(face_diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(face_thresh)

                if motion_pixels > 50:
                    face_motions[face_id] = {"last_motion": current_time, "status": "person detected"}
                elif face_id not in face_motions:
                    face_motions[face_id] = {"last_motion": current_time - PICTURE_THRESHOLD - 1, "status": "picture detected"}

    for face_id, data in face_motions.items():
        if current_time - data["last_motion"] >= PICTURE_THRESHOLD:
            data["status"] = " "
        else:
            data["status"] = " "

    gaze_history = deque(maxlen=HISTORY_LENGTH)
    looking_at_camera = False
    gaze_status = " "

    for face in faces_dlib:
        landmarks = predictor(gray, face)
        looking_at_camera, gaze_status = detect_gaze(landmarks)
        gaze_history.append(looking_at_camera)
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    looking_at_camera = sum(gaze_history) > len(gaze_history) // 2

    for i, (x, y, w, h) in enumerate(faces_cv):
        face_id = f"face_{i}"

        if face_id in face_motions:
            status = face_motions[face_id]["status"]
            color = (0, 255, 0) if status == "person detected" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if i == main_interviewer_idx:
            cv2.putText(frame, '', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif face_areas[i] < face_areas[main_interviewer_idx] * 0.6:
            cv2.putText(frame, '', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    prev_gray = gray.copy()
    
    return frame

def generate():
    global output_frame, video_lock

    while True:
        with video_lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def process_video():
    global  output_frame, video_lock, video_capture
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = process_frame(frame)

        with video_lock:
            output_frame = frame.copy()

def save_to_wav(filename):
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    if not audio_data:
        print("No audio data to save.")
        return 
    
    audio_data = np.concatenate(audio_data, axis=0)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  
        wf.setsampwidth(2)  
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def start_recording():
    global audio_stream, video_capture
    
    # Start Video Capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        threading.Thread(target=process_video).start()

    # Start Audio Capture
    audio_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=fs)
    audio_stream.start()
    print("Recording started (audio and video)")

def stop_recording():
    global audio_stream, video_capture
    
    # Stop Audio Capture
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    
    
    

    
    save_to_wav('output.mp3')
    print("Recording stopped (audio and video)")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/start')
def start_recording_route():
    start_recording()
    return jsonify({'message': 'Recording started'}), 200

@app.route('/stop')
def stop_recording_route():
    stop_recording()
    return jsonify({'message': 'Recording stopped'}), 200

@app.route('/')
def index():
    return render_template('index.html')

questions =[
   "/Users/rahul.v/Documents/Rubixe/amrutha/test/static/questions/stat1.mp3"]

@app.route('/ask_question', methods=['GET'])
def ask_question():
    # Randomly select one of the questions
    selected_question = random.choice(questions)
    # Ensure the path provided to url_for is relative to the static directory
    question_audio_url = url_for('static', filename=selected_question.replace("/Users/rahul.v/Documents/Rubixe/amrutha/test/static/", ''))
    return jsonify({"message": "Playing question", "question_audio_url": question_audio_url})



if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    t = threading.Thread(target=process_video)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port="5000", debug=True,
        threaded=True, use_reloader=False)

    video_capture.release()
