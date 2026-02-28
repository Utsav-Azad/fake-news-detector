# backend/utils/summarizer.py

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def summarize(text, num_sentences=2):
    
    # 1️⃣ Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    if len(sentences) <= num_sentences:
        return text

    # 2️⃣ TF-IDF on sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    # 3️⃣ Score sentences by sum of TF-IDF values
    scores = np.sum(sentence_vectors.toarray(), axis=1)

    # 4️⃣ Pick top sentences
    top_indices = np.argsort(scores)[-num_sentences:]
    top_indices = sorted(top_indices)

    summary = " ".join([sentences[i] for i in top_indices])

    return summary