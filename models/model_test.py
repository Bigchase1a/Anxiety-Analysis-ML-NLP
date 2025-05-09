import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from helpers.text_preprocessing import preprocess_text

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
model_path = os.path.join(base_path, "anxiety_model_20250509_1951.pkl")
vec_path = os.path.join(base_path, "tfidf_vectorizer_20250509_1951.pkl")

with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open(vec_path, "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

test_sentences = [
    "I keep thinking that maybe if I just disappeared for a while, no one would really notice.",
    "Work has been a bit busy lately, but I’m managing my time well and making progress on all my tasks without much stress.",
    "Even when I'm surrounded by people, I can't stop feeling like something terrible is going to happen, and I don’t know how to make it stop.",
    "Some days are better than others. I try to stay positive, but there are times when I feel a bit overwhelmed for no clear reason.",
    "It was a quiet afternoon. I sat under the tree reading a book, sipping tea, and listening to the wind brush the leaves.",
    "I spent the weekend with friends at the beach, laughing, swimming, and completely disconnecting from work.",
    "Lately I’ve been waking up with a tight chest and racing thoughts about things that haven’t even happened yet.",
    "I don’t think I’m depressed or anxious, but I do get tired a lot and I find it hard to stay focused, especially when deadlines approach."
]

preprocessed = [preprocess_text(text) for text in test_sentences]
vectorized = loaded_vectorizer.transform(preprocessed)
predictions = loaded_model.predict(vectorized)
probas = loaded_model.predict_proba(vectorized)

def interpret_score(score):
    if score < 0.4:
        return "Normal"
    elif score < 0.65:
        return "Muhtemel Anksiyete"
    else:
        return "Anksiyete"

for i, sentence in enumerate(test_sentences):
    anxiety_score = probas[i][1]
    label = interpret_score(anxiety_score)
    print(f"Cümle: {sentence}\n→ Tahmin: {label} ({anxiety_score * 100:.2f}%)\n")