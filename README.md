# Handwritten Notes Digitizer

The Handwritten Notes Digitizer is a web-based tool that allows users to convert handwritten notes into editable and searchable digital text using Optical Character Recognition (OCR). It provides a simple and intuitive interface for uploading images or PDFs of handwritten notes and extracting the text with the help of Python and Tesseract OCR.

---

## Features

-  Upload handwritten notes as images or PDF files
-  Extract text using Tesseract OCR
-  Organize and download digitized notes
-  User authentication system (Login/Register)
-  Save digitized notes for future access
-  Clean and beginner-friendly UI

---

##  Tech Stack

**Frontend:**
- HTML5
- CSS3
- JavaScript

**Backend:**
- Python (Flask)

**OCR Engine:**
- Tesseract OCR
- OpenCV (for image preprocessing)

**Database:**
- MySQL

**Image Storage:**
- Local storage (for now)

---

## 📁 Project Structure
handwritten-notes-digitizer/
├── static/
│ ├── css/
│ └── images/
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── register.html
│ ├── dashboard.html
├── uploads/
├── app.py
├── ocr.py
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 🔧 Prerequisites

Make sure you have Python, pip, and MySQL installed.

### 🧪 Installation

```bash
git clone https://github.com/himanshuu-g/handwritten-notes-digitizer.git
cd handwritten-notes-digitizer
pip install -r requirements.txt

 Run the App
python app.py
Go to: http://localhost:5000 in your browser.
