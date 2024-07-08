import cv2
from ultralytics import YOLOv10

# model = YOLOv10("./models/safety-helmet-det-yolov10n.pt", task="detect")
model = YOLOv10("./models/yolov10n.pt", task="detect")

cap = cv2.VideoCapture(0)
# cap.set(3, 224)

while True:
    ret, frame = cap.read()

    if ret:
        results = model.predict(frame, imgsz=224, device=0, max_det=1, conf=0.5)

        annotated_frame = results[0].plot()

        cv2.imshow("result", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
