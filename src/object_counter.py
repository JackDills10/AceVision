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
    print(counts)
    annotated_frame = results[0].plot()
    cv2.imshow("AceVision Object Counting", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()