# import cv2

# # Load the pre-trained Haar Cascade face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Open the webcam (0 is the default camera)
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     if len(faces) > 0:
#         # Calculate areas and find the index of the largest one (main interviewer)
#         face_areas = [w*h for (x, y, w, h) in faces]
#         max_index = face_areas.index(max(face_areas))
        
#         # Highlight the closest face as the "Main Interviewer"
#         x, y, w, h = faces[max_index]
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, 'Main Interviewer', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Mark other faces as "Unknown Breach" if they are distant
#         for i, (x, y, w, h) in enumerate(faces):
#             if i != max_index and face_areas[i] < face_areas[max_index] * 0.6:  # Assuming a breach if face is smaller
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, 'Unknown Breach', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
    
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import time
import threading

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
frame_skip = 5  # Process every nth frame for certain checks

# Initialize webcam
video_capture = cv2.VideoCapture(0)
frame_lock = threading.Lock()

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
    return faces, gray_image

def capture_frame():
    global current_frame
    while True:
        ret, frame = video_capture.read()
        if ret:
            with frame_lock:
                current_frame = frame

def main():
    global last_check_time, prev_gray

    gaze_history = deque(maxlen=HISTORY_LENGTH)
    frame_counter = 0
    current_frame = None

    # Start the thread for frame capturing
    thread = threading.Thread(target=capture_frame)
    thread.daemon = True
    thread.start()
    
    while True:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None

        if frame is None:
            continue

        faces_cv, gray = detect_bounding_box(frame)

        # Skip frames for certain operations to save computation
        if frame_counter % frame_skip == 0:
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
            faces_dlib = detector(gray)
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

        # Find the "Main Interviewer" (largest face)
        face_areas = [w * h for (x, y, w, h) in faces_cv]
        main_interviewer_idx = face_areas.index(max(face_areas)) if face_areas else None

        for i, (x, y, w, h) in enumerate(faces_cv):
            face_id = f"face_{i}"

            if face_id in face_motions:
                status = face_motions[face_id]["status"]
                color = (0, 255, 0) if status == "person detected" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
                cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mark "Main Interviewer" and other smaller faces as "Unknown Breach"
            if i == main_interviewer_idx:
                cv2.putText(frame, 'Main Interviewer', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif main_interviewer_idx is not None and face_areas[i] < face_areas[main_interviewer_idx] * 0.6:
                cv2.putText(frame, 'Unknown Breach', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face, Motion, and Gaze Detection", frame)

        prev_gray = gray.copy()

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

