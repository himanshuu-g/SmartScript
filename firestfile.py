import cv2
import pytesseract

# Load the image
img = cv2.imread("uploaded.jpg")

# Display the original image (optional, for debugging)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()