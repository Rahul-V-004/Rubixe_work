import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

# Load the face detection and landmark prediction models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("open_cv/shape_predictor_68_face_landmarks.dat")

# Constants for gaze detection
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
GAZE_THRESH = 2  # Threshold for determining gaze direction (in pixels)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_pupil(eye_region, frame):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def detect_gaze(landmarks, frame):
    def get_eye_points(indices):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices])

    left_eye = get_eye_points(LEFT_EYE_INDICES)
    right_eye = get_eye_points(RIGHT_EYE_INDICES)

    # Check if eyes are open
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < EYE_AR_THRESH:
        return "Eyes Closed", frame

    # Calculate center point between eyes
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    center_point = ((left_eye_center + right_eye_center) / 2).astype(int)

    # Extract eye regions
    left_eye_region = frame[left_eye_center[1]-10:left_eye_center[1]+10, left_eye_center[0]-15:left_eye_center[0]+15]
    right_eye_region = frame[right_eye_center[1]-10:right_eye_center[1]+10, right_eye_center[0]-15:right_eye_center[0]+15]

    # Detect pupils
    left_pupil = detect_pupil(left_eye_region, frame)
    right_pupil = detect_pupil(right_eye_region, frame)

    if left_pupil is None or right_pupil is None:
        return "Pupil not detected", frame

    # Calculate pupil positions relative to eye centers
    left_pupil_pos = (left_eye_center[0] - 15 + left_pupil[0], left_eye_center[1] - 10 + left_pupil[1])
    right_pupil_pos = (right_eye_center[0] - 15 + right_pupil[0], right_eye_center[1] - 10 + right_pupil[1])

    # Calculate distances from center
    left_distance = center_point[0] - left_pupil_pos[0]
    right_distance = right_pupil_pos[0] - center_point[0]

    # Visualize
    cv2.circle(frame, tuple(center_point), 3, (255, 0, 0), -1)  # Center point
    cv2.circle(frame, left_pupil_pos, 3, (0, 255, 0), -1)  # Left pupil
    cv2.circle(frame, right_pupil_pos, 3, (0, 255, 0), -1)  # Right pupil

    # Determine gaze direction
    if abs(left_distance - right_distance) < GAZE_THRESH:
        return "Center", frame
    elif left_distance > right_distance:
        return "Right", frame
    else:
        return "Left", frame

def live_gaze_analysis():
    cap = cv2.VideoCapture(0)  # Use default camera (usually webcam)
    
    total_frames = 0
    center_frames = 0
    start_time = time.time()
    duration = 120  # Run for 60 seconds

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            gaze_direction, frame = detect_gaze(landmarks, frame)
            
            if gaze_direction == "Center":
                center_frames += 1
                cv2.putText(frame, "Center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, gaze_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            total_frames += 1
        
        # Display the resulting frame
        cv2.imshow('Live Gaze Analysis', frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    percentage_looking_at_center = (center_frames / total_frames) * 100 if total_frames else 0
    return total_frames, center_frames, percentage_looking_at_center

# Main execution
total_frames, center_frames, percentage_looking_at_center = live_gaze_analysis()

print(f"\nAnalysis complete.")
print(f"Total frames: {total_frames}")
print(f"Frames looking at center: {center_frames}")
print(f"Percentage looking at center: {percentage_looking_at_center:.2f}%")
print(f"Conclusion: {'Success' if percentage_looking_at_center >= 80 else 'Failed'}")