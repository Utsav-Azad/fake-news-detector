import streamlit as st
import pickle
import os
from utils.preprocess import clean_text
from utils.credibility import calculate_credibility
from utils.summarizer import summarize

# Load model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")

st.write("Enter a news article below to check whether it is Real or Fake.")

text = st.text_area("Paste News Article Here", height=200)

if st.button("Analyze News"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])

        prediction = model.predict(vect)[0]
        probability = model.predict_proba(vect)[0]
        confidence = max(probability)

        credibility_score = calculate_credibility(text)
        summary = summarize(text)

        st.subheader("Result")

        if prediction == 1:
            st.success("Prediction: REAL")
        else:
            st.error("Prediction: FAKE")

        st.write(f"Confidence Score: {round(float(confidence), 2)}")
        st.write(f"Credibility Score: {credibility_score}/100")

        st.subheader("Summary")
        st.write(summary)