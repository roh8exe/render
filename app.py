from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

# Google Script for logging results
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxQqz_osnBNgWpWb8whnrwo9OuIxc2bh1ZlQX3VUaD8hCAdCHiI7UYzDQ0O22aPv2d9Dw/exec"

# Hugging Face API Token (Set this in Render environment variables)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

app = Flask(__name__)
CORS(app)

# Hugging Face Model API Endpoints
MODEL_ENDPOINTS = {
    "hi": "https://api-inference.huggingface.co/models/LingoIITGN/mBERT_toxic_hindi",
    "te": "https://api-inference.huggingface.co/models/LingoIITGN/mBERT_toxic_telugu"
}

# Function to send a request to Hugging Face API
def get_toxicity_prediction(text, lang):
    if lang not in MODEL_ENDPOINTS:
        return None, f"Model for language '{lang}' not found"

    url = MODEL_ENDPOINTS[lang]
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()[0]  # Extract first prediction result
        return {
            "toxicity": result["score"] * 100,  # Convert to percentage
            "is_toxic": result["label"].lower() == "toxic"
        }, None
    else:
        return None, f"Error from Hugging Face API: {response.text}"
    
@app.route('/debug-env', methods=['GET'])
def debug_env():
    return jsonify({"HF_API_TOKEN": os.environ.get("HF_API_TOKEN", "Not Found")})
    
@app.route('/')
def home():
    return "ToxiGuard API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "").strip()
    lang = data.get("lang", "hi")  # Default to Hindi if not provided

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result, error = get_toxicity_prediction(text, lang)
    if error:
        return jsonify({"error": error}), 400

    # Send results to Google Sheet
    try:
        payload = {
            "text": text,
            "lang": lang,
            "toxicity": result["toxicity"],
            "is_toxic": result["is_toxic"]
        }
        requests.post(GOOGLE_SCRIPT_URL, json=payload)
    except Exception as e:
        print(f"Error sending to Google Apps Script: {e}")

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port)
