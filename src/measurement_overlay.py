from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)
while True:
    success, frame = camera.read()
    if not success:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        width = x2 - x1
        height = y2 - y1
        text = f"{width}px x {height}px"
        cv2.putText(annotated_frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("AceVision Measurement", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()