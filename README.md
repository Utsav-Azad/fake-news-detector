# 📰 Fake News Detection System

An AI-powered web application that detects whether a news article is Real or Fake using Machine Learning and Natural Language Processing (NLP).

---

## 🚀 Live Demo

🔗 Streamlit App: [https://your-app-name.streamlit.app](https://fake-news-detector-etjxyzigygl9ortvo7q28o.streamlit.app/)  
🔗 GitHub Repository: https://github.com/your-username/fake-news-detector  

---

## 📌 Problem Statement

The rapid spread of misinformation through digital platforms makes it difficult for users to verify the authenticity of news articles. This project aims to build a machine learning system that can automatically classify news as Real or Fake and assist users in identifying misleading information.

---

## 🧠 Proposed Solution

The system uses:
- Text Preprocessing
- TF-IDF Feature Extraction
- Logistic Regression Classifier
- Credibility Scoring
- Article Summarization

Users can paste a news article and instantly receive:
- Prediction (Real/Fake)
- Confidence Score
- Credibility Score
- Summary

---

## 🏗️ System Architecture

User Input  
→ Text Preprocessing  
→ TF-IDF Vectorization  
→ Logistic Regression Model  
→ Prediction  
→ Display Results  

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLP (Natural Language Processing)

---

## 📊 Model Details

- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## 📂 Project Structure

fake-news-detector/
│
├── app_streamlit.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── utils/
    ├── preprocess.py
    ├── credibility.py
    └── summarizer.py

---

## ⚙️ Installation & Setup

1️⃣ Clone Repository

git clone https://github.com/your-username/fake-news-detector.git  
cd fake-news-detector  

2️⃣ Install Dependencies

pip install -r requirements.txt  

3️⃣ Run Application

streamlit run app_streamlit.py  

---

## 🎯 Future Scope

- Integration of Deep Learning models (BERT, LSTM)
- Multilingual support
- Real-time fact-check API integration
- Explainable AI features

---

## 👨‍💻 Author

Utsav Azad  
B.Tech – Computer Science  
Machine Learning & Full-Stack Enthusiast  

---

## 📜 License

This project is developed for academic and learning purposes.
