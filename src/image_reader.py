import cv2

image_path = "assets/images/test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image could not be loaded")
else:
    print("Image loaded successfully!")
    print("Image dimensions:", image.shape)