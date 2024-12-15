from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time
import spacy
from PIL import Image
import pytesseract
import re
import cv2
import numpy as np
from transformers import pipeline
from langdetect import detect
import langid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models and pipelines
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def advanced_preprocess_image(image_path, debug=False):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to load the image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        processed_image = Image.fromarray(morph)

        if debug:
            print("Image preprocessing successful.")
        
        return processed_image
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        language, confidence = langid.classify(text)
        return language

def categorize_text(text, debug=False):
    extracted_data = []
    phone_pattern = re.compile(r'\b010[-\s]?\d{4}[-\s]?\d{4}\b')
    data = {}

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        phone_match = re.search(phone_pattern, line)
        if phone_match:
            data["Phone Number"] = phone_match.group()

        if "\uac01\uc7a5" in line or "\uad6c" in line or "\ub3d9" in line:
            data["Address"] = line
        elif len(line.split()) == 1 and "Name" not in data:
            data["Name"] = line

        if "\uad11\uace0" in line:
            data["Type of Information"] = "\uad11\uace0"
        elif "\uc9c0\uc815\ub418\uc9c0 \uc54a\uc74c" in line:
            data["Type of Information"] = "\uc9c0\uc815\ub418\uc9c0 \uc54a\uc74c"

        sentiment = sentiment_analyzer(line)
        if sentiment and sentiment[0]["label"] == "POSITIVE":
            data["Rating"] = 10.00
        elif "Rating" not in data:
            data["Rating"] = 9.83

        if all(key in data for key in ["Name", "Address", "Phone Number", "Type of Information", "Rating"]):
            extracted_data.append(data)
            data = {}

    return extracted_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Perform OCR on the original image
                original_image = Image.open(filepath)
                ocr_text = pytesseract.image_to_string(original_image, lang="kor", config='--psm 6')

                if not ocr_text.strip():
                    # Try advanced preprocessing if initial OCR fails
                    processed_image = advanced_preprocess_image(filepath)
                    if processed_image:
                        ocr_text = pytesseract.image_to_string(processed_image, lang="kor", config='--psm 6')

                if ocr_text.strip():
                    # Detect language and categorize text
                    language = detect_language(ocr_text)
                    data = categorize_text(ocr_text)
                    
                    # Clean up uploaded file
                    os.remove(filepath)
                    
                    return render_template('index.html', data=data, success=True)
                else:
                    return render_template('index.html', error="No text could be extracted from the image")

            except Exception as e:
                return render_template('index.html', error=f"Error processing image: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
