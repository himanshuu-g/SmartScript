import os
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image, ImageEnhance
import re
from statistics import mean
from difflib import SequenceMatcher

# Set path to Tesseract (only needed on Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input file (can be image or PDF)
file_path = "uploaded.pdf"  # or "uploaded.jpg"

# Create a debug directory for intermediate processing images
debug_dir = "debug_images"
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

def create_processing_variants(image):
    """Create multiple processing variants of the same image"""
    variants = []
    
    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Save original grayscale
    variants.append(("original_gray", gray))
    
    # VARIANT 1: Adaptive Threshold with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    variant1 = cv2.bitwise_not(closed)
    variants.append(("adaptive_clahe", variant1))
    
    # VARIANT 2: Otsu threshold with bilateral filtering
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    variant2 = cv2.bitwise_not(closed)
    variants.append(("otsu_bilateral", variant2))
    
    # VARIANT 3: High contrast with Gaussian blur
    pil_img = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_img)
    high_contrast = np.array(enhancer.enhance(2.5))
    blurred = cv2.GaussianBlur(high_contrast, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    variant3 = cv2.bitwise_not(cleaned)
    variants.append(("high_contrast", variant3))
    
    # VARIANT 4: Scaled image (2x upscaling often helps Tesseract)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, scaled_thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scaled_thresh = cv2.bitwise_not(scaled_thresh)
    variants.append(("scaled_2x", scaled_thresh))
    
    # VARIANT 5: Binarization with Niblack's method
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Implement a simplified version of Niblack's method
    window_size = 25
    k = -0.2
    mean = cv2.boxFilter(denoised, -1, (window_size, window_size), normalize=True)
    mean_square = cv2.boxFilter(denoised**2, -1, (window_size, window_size), normalize=True)
    variance = mean_square - mean**2
    std = np.sqrt(variance)
    threshold = mean + k * std
    binary = np.zeros_like(denoised)
    binary[denoised > threshold] = 255
    variants.append(("niblack", binary))
    
    return variants

def find_best_psm_for_image(image):
    """Try different PSM modes and find best one based on confidence score"""
    psm_modes = [6, 3, 4, 11, 1]  # Different segmentation modes
    best_psm = 6  # Default
    best_confidence = 0
    
    for psm in psm_modes:
        try:
            config = f'--oem 1 --psm {psm} -l eng --dpi 300'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            if len(data['conf']) > 0:
                # Filter out -1 confidence values
                confidences = [float(conf) for conf in data['conf'] if conf != -1]
                if confidences:
                    avg_confidence = mean(confidences)
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_psm = psm
        except:
            continue
    
    return best_psm

def get_text_with_confidence(image, psm=6):
    """Get OCR text with confidence metrics"""
    config = f'--oem 1 --psm {psm} -l eng --dpi 300'
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    
    # Extract words and their confidence levels
    words = []
    confidences = []
    
    for i in range(len(data['text'])):
        if data['text'][i].strip() and float(data['conf'][i]) > -1:  # Only include non-empty text
            words.append(data['text'][i])
            confidences.append(float(data['conf'][i]))
    
    # Combine words into text
    text = ' '.join(words)
    avg_confidence = mean(confidences) if confidences else 0
    
    return text, avg_confidence

def perform_ocr_with_variants(image_path, is_temp=False):
    """Perform OCR using multiple processing variants and combine results"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Could not read image at {image_path}"
    
    # Try different image processing variants
    variants = create_processing_variants(image)
    
    # Try OCR on each variant
    results = []
    for name, processed_image in variants:
        # Save processed image for debugging
        cv2.imwrite(os.path.join(debug_dir, f"{name}_{os.path.basename(image_path)}"), processed_image)
        
        # Find best PSM mode for this variant
        best_psm = find_best_psm_for_image(processed_image)
        
        # Get text with confidence
        text, confidence = get_text_with_confidence(processed_image, best_psm)
        
        # Try with explicit character whitelist if result is poor
        if confidence < 60:
            whitelist_config = f'--oem 1 --psm {best_psm} -l eng --dpi 300 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:\'\"!?-+=/\\()\n "'
            text_whitelist = pytesseract.image_to_string(processed_image, config=whitelist_config)
            results.append((text_whitelist, confidence, name))
        
        # Store results
        results.append((text, confidence, name))
    
    # Also try with original image at different scales
    for scale in [1.5, 2.0, 0.75]:
        scaled_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if len(scaled_img.shape) == 3:
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        
        # Try different engines for scaled image
        for oem in [1, 3]:  # LSTM only or both LSTM and legacy
            best_psm = find_best_psm_for_image(scaled_img)
            config = f'--oem {oem} --psm {best_psm} -l eng --dpi 300'
            text = pytesseract.image_to_string(scaled_img, config=config)
            
            # Use simplified confidence (length-based)
            confidence = len(text.strip())
            results.append((text, confidence, f"scale_{scale}_oem_{oem}"))
    
    # Find best result based on confidence and content
    if not results:
        return "No OCR results obtained"
    
    # Sort by confidence (higher is better)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 3 results for merging
    top_results = results[:3]
    
    # Clean up temp files if needed
    if is_temp and os.path.exists(image_path):
        os.remove(image_path)
    
    # Merge results using some heuristics (choose most common words)
    merged_text = merge_ocr_results([r[0] for r in top_results])
    
    return merged_text

def merge_ocr_results(texts):
    """Merge multiple OCR results to get best possible text"""
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    # Split texts into words
    word_sets = [re.findall(r'\b\w+\b', text.lower()) for text in texts]
    
    # Get the most common words across all results
    word_counts = {}
    for word_set in word_sets:
        for word in word_set:
            if len(word) > 1:  # Ignore single characters, likely noise
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Find the text with the most common words
    best_text_index = 0
    best_common_count = 0
    
    for i, word_set in enumerate(word_sets):
        common_count = sum(word_counts.get(word, 0) for word in word_set)
        if common_count > best_common_count:
            best_common_count = common_count
            best_text_index = i
    
    # Return the best text
    return texts[best_text_index]

def process_pdf(pdf_path, dpi=300):
    """Process PDF file page by page with higher DPI"""
    all_text = ""
    
    try:
        # Convert PDF to images with higher DPI for better quality
        print(f"Converting PDF to images at {dpi} DPI...")
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        for page_num, page in enumerate(pages):
            print(f"Processing page {page_num + 1}/{len(pages)}...")
            
            # Save page as temporary image
            temp_image_path = f"temp_page_{page_num + 1}.png"  # PNG for lossless quality
            page.save(temp_image_path, "PNG")
            
            # Process the page
            text = perform_ocr_with_variants(temp_image_path, is_temp=True)
            all_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
    
    except Exception as e:
        return f"Error processing PDF: {str(e)}"
    
    return all_text

def post_process_text(text):
    """Clean up the extracted text"""
    # Replace common OCR errors
    replacements = {
        '|': 'I',
        '0': 'O',
        '1': 'I',
        'rn': 'm',
        'vv': 'w'
    }
    
    cleaned_text = text
    for old, new in replacements.items():
        cleaned_text = cleaned_text.replace(old, new)
    
    # Normalize spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Fix sentence spacing
    cleaned_text = re.sub(r'(\. ?)([A-Z])', r'. \2', cleaned_text)
    
    # Remove excess newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

def create_handwriting_processing_variants(image):
    """Create processing variants specifically for handwritten notes"""
    variants = []

    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Save original grayscale
    variants.append(("original_gray", gray))
    
    # VARIANT 1: Adaptive Threshold with CLAHE (Improved for handwritten text)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Stronger CLAHE for better contrast
    enhanced = clahe.apply(gray)
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    variant1 = cv2.bitwise_not(closed)
    variants.append(("adaptive_clahe", variant1))

    # VARIANT 2: Denoising and Adaptive Thresholding
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    variant2 = cv2.bitwise_not(closed)
    variants.append(("denoised_otsu", variant2))

    # VARIANT 3: High contrast with Gaussian blur (helps for rough handwriting)
    pil_img = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_img)
    high_contrast = np.array(enhancer.enhance(2.0))
    blurred = cv2.GaussianBlur(high_contrast, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    variant3 = cv2.bitwise_not(cleaned)
    variants.append(("high_contrast", variant3))

    return variants


def get_text_with_handwriting_config(image):
    """Apply OCR with custom configuration for handwritten notes"""
    config = '--oem 1 --psm 6 -l eng --dpi 300'  # Using LSTM only OCR engine for handwriting
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    
    words = []
    confidences = []
    
    for i in range(len(data['text'])):
        if data['text'][i].strip() and float(data['conf'][i]) > -1:
            words.append(data['text'][i])
            confidences.append(float(data['conf'][i]))
    
    text = ' '.join(words)
    avg_confidence = mean(confidences) if confidences else 0
    
    return text, avg_confidence


def perform_ocr_on_handwritten(image_path):
    """Perform OCR specifically for handwritten notes with improved processing"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Could not read image at {image_path}"

    # Try different image processing variants
    variants = create_handwriting_processing_variants(image)
    
    # OCR results list
    results = []
    
    # Try OCR on each variant
    for name, processed_image in variants:
        # Save processed image for debugging
        cv2.imwrite(os.path.join(debug_dir, f"{name}_{os.path.basename(image_path)}"), processed_image)
        
        # Get text with confidence
        text, confidence = get_text_with_handwriting_config(processed_image)
        
        # Add result
        results.append((text, confidence, name))
    
    # Sort results by confidence and choose the top one
    results.sort(key=lambda x: x[1], reverse=True)
    best_text = results[0][0] if results else ""
    
    return best_text


# Example usage for handwritten notes
image_path = "handwritten_note.jpg"  # Path to handwritten note image
extracted_text = perform_ocr_on_handwritten(image_path)
print(extracted_text)

# Main execution
if __name__ == "__main__":
    extracted_text = ""

    try:
        print(f"Processing file: {file_path}")
        
        if file_path.lower().endswith(".pdf"):
            print("Detected PDF file, processing page by page...")
            extracted_text = process_pdf(file_path)
        else:
            print("Processing single image file...")
            extracted_text = perform_ocr_with_variants(file_path)
        
        # Clean up the extracted text
        extracted_text = post_process_text(extracted_text)
        
        # Save the extracted text to a file
        with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        
        print("\n📝 Extraction complete! Text saved to 'extracted_text.txt'")
        print("\nFirst 500 characters of extracted text:")
        print(extracted_text[:500] + "...")


    except Exception as e:
        print(f"Error: {str(e)}")