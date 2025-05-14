import cv2
import pytesseract

# Set Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image = cv2.imread("uploaded.jpg")

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract text
extracted_text = pytesseract.image_to_string(gray)

# Save text to a file
with open("extracted_notes.txt", "w", encoding="utf-8") as file:
    file.write(extracted_text)

print("✅ Notes saved in 'extracted_notes.txt'")
