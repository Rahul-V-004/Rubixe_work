import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import time

# Load OpenCV's Haar Cascade face detector and dlib's facial landmark predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("open_cv/shape_predictor_68_face_landmarks.dat")

# Constants for gaze detection
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
CHECK_INTERVAL = 3  # Check for motion every 3 seconds
PICTURE_THRESHOLD = 5  # Declare as picture if no motion for 5 seconds

# Initialize variables for motion detection
prev_gray = None
face_motions = {}
last_check_time = time.time()

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Utility functions for eye aspect ratio and gaze detection
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

def main():
    global last_check_time

    gaze_history = deque(maxlen=HISTORY_LENGTH)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_cv = detect_bounding_box(frame)
        faces_dlib = detector(gray)
               

        current_time = time.time()

        # Motion detection logic
        if current_time - last_check_time >= CHECK_INTERVAL:
            last_check_time = current_time

            if prev_gray is not None and len(faces_cv) > 0:
                for i, (x, y, w, h) in enumerate(faces_cv):
                    face_id = f"face_{i}"

                    face_diff = cv2.absdiff(prev_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                    _, face_thresh = cv2.threshold(face_diff, 25, 255, cv2.THRESH_BINARY)
                    motion_pixels = cv2.countNonZero(face_thresh)

                    if motion_pixels > 50:  # Motion detected
                        face_motions[face_id] = {"last_motion": current_time, "status": "person detected"}
                    elif face_id not in face_motions:
                        face_motions[face_id] = {"last_motion": current_time - PICTURE_THRESHOLD - 1, "status": "picture detected"}

        for face_id, data in face_motions.items():
            if current_time - data["last_motion"] >= PICTURE_THRESHOLD:
                data["status"] = "picture detected"
            else:
                data["status"] = "person detected"

        # Gaze detection logic
        looking_at_camera = False
        gaze_status = "No face detected"
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

        cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face, Motion, and Gaze Detection", frame)

        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()