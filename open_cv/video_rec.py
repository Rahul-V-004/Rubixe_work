import cv2
import time

# Set up video capture
video_capture = cv2.VideoCapture(0)
output_filename = 'recorded_video.avi'
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# Record for a specific amount of time (e.g., 30 seconds)
record_time = 30  # in seconds
start_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Show the recording indicator
    time_elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Recording... {time_elapsed}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output file
    output.write(frame)
    cv2.imshow("Recording Video", frame)

    # Check if recording time is over
    if time.time() - start_time > record_time:
        print("Recording completed.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
output.release()
cv2.destroyAllWindows()
