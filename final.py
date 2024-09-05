import cv2
import numpy as np
import time

# Load the face detection classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize variables for motion detection
prev_gray = None
face_motions = {}
last_check_time = time.time()
check_interval = 3  # Check for motion every 3 seconds
picture_threshold = 5  # Declare as picture if no motion for 5 seconds

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces

while True:
    # Capture the current frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces in the current frame
    faces = detect_bounding_box(frame)

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()

    # Check for motion every 3 seconds
    if current_time - last_check_time >= check_interval:
        last_check_time = current_time

        if prev_gray is not None and len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                face_id = f"face_{i}"
                
                # Compute the absolute difference between the current and previous frame
                # but only within the face bounding box
                face_diff = cv2.absdiff(
                    prev_gray[y:y+h, x:x+w],
                    gray[y:y+h, x:x+w]
                )

                # Threshold the difference
                _, face_thresh = cv2.threshold(face_diff, 25, 255, cv2.THRESH_BINARY)

                # Count non-zero pixels (motion pixels)
                motion_pixels = cv2.countNonZero(face_thresh)

                # If the number of motion pixels is above a threshold, consider it as motion
                if motion_pixels > 50:  # You can adjust this threshold
                    face_motions[face_id] = {"last_motion": current_time, "status": "person detected"}
                elif face_id not in face_motions:
                    face_motions[face_id] = {"last_motion": current_time - picture_threshold - 1, "status": "picture detected"}

    # Update status for each face
    for face_id, data in face_motions.items():
        if current_time - data["last_motion"] >= picture_threshold:
            data["status"] = "picture detected"
        else:
            data["status"] = "person detected"

    # Draw rectangles around faces and display status
    for i, (x, y, w, h) in enumerate(faces):
        face_id = f"face_{i}"
        if face_id in face_motions:
            status = face_motions[face_id]["status"]
            color = (0, 255, 0) if status == "person detected" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Face and Motion Detection", frame)

    # Update the previous frame
    prev_gray = gray.copy()

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()