import cv2
from ultralytics import YOLO

# YOLO modelini yükleyin (YOLOv8 modelini kullanıyoruz)
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' YOLOv8'in en küçük versiyonu

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan kareyi al
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # YOLO modelini kullanarak objeleri tespit et
    results = model(frame)

    # Sonuçları çizin
    annotated_frame = results[0].plot()  # Tespit edilen nesneleri çizer

    # Ekranda göster
    cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

    # 'q' tuşuna basılırsa döngüyü kır ve çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
