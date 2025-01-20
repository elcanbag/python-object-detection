import cv2
from ultralytics import YOLO

# YOLO modelini yükleyin
model = YOLO("../yolov8n.pt")

# Video dosyasını yükleyin
# video_path = "videos/road.mp4"
# video_path = "videos/fpvdrone.mp4"
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

    # Video çıkışı oluştur
    output = cv2.VideoWriter(
        "../output_video.avi",  # AVI formatında kaydedin
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_width, frame_height)
    )

    frame_skip = 5  # Her 5 karede bir analiz yap
    frame_count = 0  # Kare sayacını başlat
    previous_results = None  # Önceki tespit sonuçlarını sakla

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Kare sayısını artır
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        if frame_count % frame_skip == 0:
            # YOLO ile tespit yap
            results = model(resized_frame)
            previous_results = results  # Sonuçları sakla
        else:
            results = previous_results  # Önceki sonuçları kullan

        # Kareyi çizin
        if results:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = resized_frame

        # İşlenmiş kareyi kaydedin
        output.write(annotated_frame)

        # Kareyi ekranda göster
        cv2.imshow("YOLOv8 Object Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
