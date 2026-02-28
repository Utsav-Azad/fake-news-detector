# backend/models/train_model.py

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocess import clean_text


# -----------------------------
# 1️⃣ Setup Base Path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")

# -----------------------------
# 2️⃣ Load Dataset
# -----------------------------
fake = pd.read_csv(os.path.join(data_path, "Fake.csv"))
true = pd.read_csv(os.path.join(data_path, "True.csv"))

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge
df = pd.concat([fake, true], axis=0)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create content column
df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]

# -----------------------------
# 3️⃣ Preprocess Text
# -----------------------------
df["content"] = df["content"].apply(clean_text)

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 5️⃣ TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6️⃣ Train Model
# -----------------------------
model = LogisticRegression(max_iter=100000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 7️⃣ Evaluate
# -----------------------------
predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -----------------------------
# 8️⃣ Save Model & Vectorizer
# -----------------------------
pickle.dump(model, open(os.path.join(BASE_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))

print("\nModel and Vectorizer saved successfully! 🚀")