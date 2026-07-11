from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)
while True:
    success, frame = camera.read()
    if not success:
        break
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    cv2.imshow("AceVision Object Tracking", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()