import cv2

input_path = "assets/images/test.jpeg"
output_path = "assets/images/processed_test.jpg"

image = cv2.imread(input_path)

if image is None:
    print("Could not load image")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_path, gray_image)

    print("Image processed successfully!")
    print("Saved as:", output_path)