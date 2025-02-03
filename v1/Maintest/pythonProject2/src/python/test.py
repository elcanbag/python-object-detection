import cv2
from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO("../yolov8n.pt")

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_folder = "cameraoutput"
os.makedirs(output_folder, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_folder, f"output_{current_time}.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera is not worked.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    video_writer.write(annotated_frame)

    cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video was saved: {output_path}")
