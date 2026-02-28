# backend/utils/credibility.py

import re

def calculate_credibility(text):
    score = 100
    text_lower = text.lower()

    # 1️⃣ Too many ALL CAPS words
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
    if len(caps_words) > 3:
        score -= 15

    # 2️⃣ Excessive exclamation marks
    if text.count("!") > 3:
        score -= 10

    # 3️⃣ Clickbait phrases
    clickbait_words = [
        "shocking",
        "breaking",
        "you won't believe",
        "what happens next",
        "secret revealed",
        "must see"
    ]

    for phrase in clickbait_words:
        if phrase in text_lower:
            score -= 15

    # 4️⃣ Very short article (suspicious)
    if len(text.split()) < 50:
        score -= 10

    # Ensure score stays between 0–100
    return max(min(score, 100), 0)