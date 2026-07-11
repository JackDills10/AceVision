from ultralytics import YOLO
model = YOLO("yolo11n.pt")
image_path = "assets/images/test.jpeg"
results = model(image_path)
for result in results:
    annotated_image = result.plot()
    output_path = "assets/images/detection_result.jpg"
    result.save(filename=output_path)
    print("Detection image saved!")