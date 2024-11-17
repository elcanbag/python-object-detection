import cv2
from ultralytics import YOLO

# YOLO modelini yükleyin (YOLOv8 modelini kullanıyoruz)
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' YOLOv8'in en küçük versiyonu

# Görüntü dosyasını yükleyin
image_path = "images/road.png"  # Dosya yolu
frame = cv2.imread(image_path)

if frame is None:
    print("Görüntü dosyası yüklenemedi. Lütfen dosya yolunu kontrol edin.")
else:
    # Görüntüyü yeniden boyutlandır
    scale_percent = 50  # Görüntüyü %50 oranında küçült
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # YOLO modelini kullanarak objeleri tespit et
    results = model(resized_frame)

    # Sonuçları çizin
    annotated_frame = results[0].plot()  # Tespit edilen nesneleri çizer

    # Ekranda göster
    cv2.imshow("YOLOv8 Object Detection - Image", annotated_frame)
    cv2.waitKey(0)  # Bir tuşa basılmasını bekler
    cv2.destroyAllWindows()
