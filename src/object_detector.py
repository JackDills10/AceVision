from ultralytics import YOLO
model = YOLO("yolo11n.pt")
image_path = "assets/images/test.jpeg"
results = model(image_path)
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]
        print(f"Detected: {label} | Confidence: {confidence:.2f}")