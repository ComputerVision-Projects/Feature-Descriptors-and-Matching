import cv2
from HarrisCorner import HarrisCorner  

# Load and preprocess an image
image = cv2.imread(r"D:\SBME 2026\(3rd year 2nd term) Sixth Term\computer vision\Projects\Edge-and-boundary-detection-Hough-transform-and-SNAKE-\Images\green.jpg")  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the Harris corner detector
detector = HarrisCorner(gray)

# Run the algorithm
detector.compute_harris_response()
detector.threshold_and_nms()
result_image = detector.draw_corners_on_image()

# Show the original and result images
cv2.imshow("Original", image)
cv2.imshow("Corners Detected", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
