import os
import cv2
import pytesseract

# Set Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image = cv2.imread("uploaded.jpg")

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract text
extracted_text = pytesseract.image_to_string(gray).lower()  # Convert to lowercase for matching

# Define categories and keywords
categories = {
    "math": ["math", "equation", "algebra", "geometry"],
    "physics": ["physics", "force", "energy", "motion"],
    "history": ["history", "date", "war", "king"],
}

# Default category
category = "general"

# Find the category based on keywords
for cat, keywords in categories.items():
    if any(word in extracted_text for word in keywords):
        category = cat
        break

# Create folder if it doesn't exist
if not os.path.exists(category):
    os.makedirs(category)

# Save file in the categorized folder
file_path = os.path.join(category, "notes.txt")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(extracted_text)

print(f"✅ Notes saved in '{category}/notes.txt'")
