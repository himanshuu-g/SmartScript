from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from functools import wraps
import mysql.connector
from mysql.connector import Error
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure random key in production

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'transkribus_db',
    'user': 'root',
    'password': 'newpassword'  # Replace with your MySQL password
}

# Database connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Initialize database and create tables if they don't exist
def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                country VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully")
    else:
        print("Failed to initialize database")

# Initialize the database when app starts
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Password validation
def is_valid_password(password):
    if len(password) < 8:
        return False
    # Check if password has at least 8 chars with a number and lowercase letter
    return bool(re.search(r'[0-9]', password) and re.search(r'[a-z]', password))

# Username validation
def is_valid_username(username):
    # Only alphanumeric characters or single hyphens, can't begin or end with hyphen
    pattern = r'^[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*$'
    return bool(re.match(pattern, username))

# Home page - allow access regardless of login status
@app.route("/")
def home():
    return render_template("index.html", username=session.get('username'))

# Upload Page - protected
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # OCR extraction
            text = extract_text(filepath)
            return render_template("result.html", extracted_text=text)
    return render_template("upload.html", username=session.get('username'))

# About Page
@app.route("/about")
def about():
    return render_template("about.html", username=session.get('username'))

# Contact us
@app.route("/contact")
def contact():
    return render_template("contact.html",username=session.get('username'))

# Signup page and handler
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'username' in session:
        return redirect(url_for('upload'))
        
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']
        country = request.form['country']
        
        # Validate inputs
        if not is_valid_password(password):
            flash('Password should be at least 8 characters OR at least 8 characters including a number and a lowercase letter.')
            return redirect(url_for('signup'))
            
        if not is_valid_username(username):
            flash('Username may only contain alphanumeric characters or single hyphens, and cannot begin or end with a hyphen.')
            return redirect(url_for('signup'))
        
        # Try to insert new user
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (email, username, password, country) VALUES (%s, %s, %s, %s)",
                    (email, username, password, country)
                )
                conn.commit()
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
            except Error as e:
                conn.rollback()
                if "Duplicate entry" in str(e):
                    if "email" in str(e):
                        flash('Email already exists')
                    elif "username" in str(e):
                        flash('Username already exists')
                    else:
                        flash('Registration failed. Please try again.')
                else:
                    flash(f'Registration failed: {str(e)}')
            finally:
                cursor.close()
                conn.close()
        else:
            flash('Database connection error. Please try again later.')
            
    return render_template("signup.html")

# Login page and handler
@app.route("/login", methods=["GET", "POST"])
def login():
    if 'username' in session:
        return redirect(url_for('upload'))
    
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT * FROM users WHERE username = %s AND password = %s",
                    (username, password)
                )
                user = cursor.fetchone()
                
                if user:
                    session['username'] = username
                    session['user_id'] = user['id']
                    session['email'] = user['email']
                    flash('Login successful!')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid username or password')
            finally:
                cursor.close()
                conn.close()
        else:
            flash('Database connection error. Please try again later.')
            
    return render_template("login.html")

# Logout handler
@app.route("/logout")
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('email', None)
    flash('You have been logged out')
    return redirect(url_for('home'))

# OCR Function
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            # Convert PDF to images and perform OCR
            pages = convert_from_path(file_path, 300)
            text = ""
            for page in pages:
                text += pytesseract.image_to_string(page)
            return text.strip()
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Perform OCR directly on image
            img = Image.open(file_path)
            return pytesseract.image_to_string(img).strip()
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error extracting text: {e}"


def test_db_connection():
    conn = get_db_connection()
    if conn:
        print("Database connection successful!")
        conn.close()
        return True
    else:
        print("Failed to connect to database!")
        return False

# Call this function before running the app
test_db_connection()

if __name__ == "__main__":
    app.run(debug=True)