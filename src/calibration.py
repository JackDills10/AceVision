import cv2
import numpy as np

# Known size of a US quarter
QUARTER_DIAMETER_INCHES = 0.955

image = cv2.imread("assets/images/calibration_quarter.jpeg")

if image is None:
    print("Could not load image")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=200,
    param1=100,
    param2=60,
    minRadius=50,
    maxRadius=150
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    largest_circle = max(circles[0], key=lambda c: c[2])

    x, y, radius = largest_circle

    diameter_pixels = radius * 2

    pixels_per_inch = (
        diameter_pixels / QUARTER_DIAMETER_INCHES
        )

    print(f"Quarter detected")
    print(f"Diameter: {diameter_pixels}px")
    print(f"Scale: {pixels_per_inch:.2f} pixels/inch")

    cv2.circle(
        image,
        (x, y),
        radius,
        (0, 255, 0),
        3
    )

    cv2.putText(
        image,
        f"Scale: {pixels_per_inch:.2f} px/in",
        (x - 50, y - radius - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

else:
    print("Quarter not detected")

cv2.imshow("AceVision Calibration", image)

cv2.waitKey(0)
cv2.destroyAllWindows()