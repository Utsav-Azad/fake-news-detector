# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

from utils.preprocess import clean_text
from utils.credibility import calculate_credibility
from utils.summarizer import summarize


# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__)
CORS(app)  # Allow React frontend to connect


# -------------------------
# Load Model & Vectorizer
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API Running 🚀"})


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    user_text = data.get("text", "")

    if user_text.strip() == "":
        return jsonify({"error": "No text provided"}), 400

    # 1️⃣ Preprocess
    cleaned_text = clean_text(user_text)

    # 2️⃣ Vectorize
    vect = vectorizer.transform([cleaned_text])

    # 3️⃣ Predict
    prediction = model.predict(vect)[0]
    probability = model.predict_proba(vect)[0]
    confidence = max(probability)

    # 4️⃣ Credibility Score
    credibility_score = calculate_credibility(user_text)

    # 5️⃣ Summary
    summary = summarize(user_text)

    # 6️⃣ Response
    result = {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": round(float(confidence), 2),
        "credibility_score": credibility_score,
        "summary": summary
    }

    return jsonify(result)


# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)