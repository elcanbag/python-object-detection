import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../yolov8n.pt")


#
# Sabina, please load the video files.
#


# video_path = "../videos/road.mp4"
video_path = "../videos/fpvdrone.mp4"
#video_path = "../videos/qazwsx.mp4"




cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to load the video file.")
else:
    # Resolution settings
    scale_percent = 50
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video output
    output = cv2.VideoWriter(
        "../output_video.avi",  # Save in AVI format
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_width, frame_height)
    )

    frame_skip = 5  # Perform detection every 5 frames
    frame_count = 0  # Initialize frame counter
    previous_results = None  # Store previous detection results

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment frame count
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        if frame_count % frame_skip == 0:
            # Perform detection with YOLO
            results = model(resized_frame)
            previous_results = results  # Store results
        else:
            results = previous_results  # Use previous results

        # Draw the frame
        if results:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = resized_frame

        # Save the processed frame
        output.write(annotated_frame)

        # Display the frame on screen
        cv2.imshow("YOLOv8 Object Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
