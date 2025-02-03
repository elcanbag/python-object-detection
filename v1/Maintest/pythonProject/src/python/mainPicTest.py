import cv2
from ultralytics import YOLO

# Load the YOLO model (Using YOLOv8 model)
model = YOLO("../yolov8n.pt")  # 'yolov8n.pt' is the smallest version of YOLOv8

# Load the image file
image_path = "../images/road.png"  # File path
frame = cv2.imread(image_path)

if frame is None:
    print("Failed to load the image file. Please check the file path.")
else:
    # Resize the image
    scale_percent = 50  # Reduce the image size by 50%
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Detect objects using the YOLO model
    results = model(resized_frame)

    # Draw the results
    annotated_frame = results[0].plot()  # Draw detected objects

    # Display the result
    cv2.imshow("YOLOv8 Object Detection - Image", annotated_frame)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()
