from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import os
import requests

# Google Script for logging results
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxQqz_osnBNgWpWb8whnrwo9OuIxc2bh1ZlQX3VUaD8hCAdCHiI7UYzDQ0O22aPv2d9Dw/exec"

# Hugging Face API Token (Set this in Render environment variables)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

app = Flask(__name__)
CORS(app)

# Define Model Classes
class ToxicityModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=HF_API_TOKEN).to(self.device)
        self.classifier = pipeline("text-classification", model=model_name, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def predict(self, text):
        result = self.classifier(text)[0]  # Get first result
        return {
            "toxicity": result["score"] * 100,  # Convert to percentage
            "is_toxic": result["label"].lower() == "toxic"
        }

# Load models from Hugging Face
models = {
    "hi": ToxicityModel("LingoIITGN/mBERT_toxic_hindi"),
    "te": ToxicityModel("LingoIITGN/mBERT_toxic_telugul")
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "").strip()
    lang = data.get("lang", "hi")  # Default to Hindi if not provided

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if lang not in models:
        return jsonify({"error": f"Model for language '{lang}' not found"}), 400

    result = models[lang].predict(text)

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
    app.run(host='0.0.0.0', port=2026)
