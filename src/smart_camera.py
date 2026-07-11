from ultralytics import YOLO
import cv2
from collections import Counter
model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)
while True:
    success, frame = camera.read()
    if not success:
        break
    results = model.track(frame, persist=True)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detected_objects.append(label)

    counts = Counter(detected_objects)
    annotated_frame = results[0].plot()
    y = 30
    for item, amount in counts.items():
        text = f"{item}: {amount}"
        cv2.putText(annotated_frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y += 40

    cv2.imshow("AceVision Smart Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()