import cv2
import torch
from ultralytics import YOLO

# YOLO modelini yükleyin
model = YOLO("../yolov8n.pt")

# Modelin isimleri üzerinden "train" sınıfının id'sini belirleyin
train_class_id = None
for key, value in model.names.items():
    if value.lower() == "train":
        train_class_id = int(key)
        break

if train_class_id is None:
    print("Modelde 'train' sınıfı bulunamadı.")

# Video dosyasını yükleyin
video_path = "../videos/qazwsx.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video dosyası yüklenemedi.")
else:
    # Çözünürlük ayarları
    scale_percent = 50
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video çıktı dosyasını oluşturun
    output = cv2.VideoWriter(
        "../output_video.avi",  # AVI formatında kaydediliyor
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_width, frame_height)
    )

    frame_skip = 5  # Her 5 karede bir tespit yap
    frame_count = 0  # Kare sayacı
    previous_results = None  # Önceki tespit sonuçlarını saklamak için

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Kare sayısını arttırın
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        if frame_count % frame_skip == 0:
            # YOLO ile tespit yap
            results = model(resized_frame)

            # Tespit sonuçlarında 'train' nesnesini filtreleyin
            for result in results:
                if result.boxes is not None and len(result.boxes.data) > 0:
                    boxes_data = result.boxes.data
                    # Sadece 'train' olmayan kutuları tut
                    keep = (boxes_data[:, 5].int() != train_class_id)
                    result.boxes.data = boxes_data[keep]

            previous_results = results
        else:
            results = previous_results

        # Kare üzerine tespitleri çizdir
        if results:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = resized_frame

        # İşlenmiş kareyi kaydet
        output.write(annotated_frame)

        # Kareyi ekranda göster
        cv2.imshow("YOLOv8 Object Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
