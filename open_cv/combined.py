import cv2
import dlib
import numpy as np
import time
from collections import deque
from scipy.spatial import distance as dist

# Load detectors and predictors
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("open_cv/shape_predictor_68_face_landmarks.dat")

# Constants
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5
EYE_AR_THRESH = 0.2
CHECK_INTERVAL = 3
PICTURE_THRESHOLD = 5

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

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

def main():
    cap = cv2.VideoCapture(0)
    prev_gray = None
    face_motions = {}
    last_check_time = time.time()
    gaze_history = deque(maxlen=HISTORY_LENGTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        current_time = time.time()

        if current_time - last_check_time >= CHECK_INTERVAL:
            last_check_time = current_time

            if prev_gray is not None and len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
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
                data["status"] = "picture detected"
            else:
                data["status"] = "person detected"

        dlib_faces = detector(gray)
        for i, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{i}"
            status = face_motions.get(face_id, {}).get("status", "Unknown")
            
            if status == "person detected" and i < len(dlib_faces):
                landmarks = predictor(gray, dlib_faces[i])
                looking_at_camera, gaze_status = detect_gaze(landmarks)
                gaze_history.append(looking_at_camera)
                
                looking_at_camera = sum(gaze_history) > len(gaze_history) // 2
                status += f" - {gaze_status}"
            
            color = (0, 255, 0) if "person detected" in status else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Integrated Face Analysis", frame)
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()