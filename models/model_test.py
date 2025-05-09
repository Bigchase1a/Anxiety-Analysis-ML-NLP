import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from helpers.text_preprocessing import preprocess_text

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
model_path = os.path.join(base_path, "anxiety_model_20250509_1749.pkl")
vec_path = os.path.join(base_path, "tfidf_vectorizer_20250509_1749.pkl")

with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open(vec_path, "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

test_sentences = [
    "I feel overwhelmed and can't sleep at night.",
    "Today was a calm and peaceful day.",
    "I can't stop thinking about bad things happening.",
    "I'm just chilling and watching TV."
]

preprocessed = [preprocess_text(text) for text in test_sentences]
vectorized = loaded_vectorizer.transform(preprocessed)
predictions = loaded_model.predict(vectorized)

for sentence in preprocessed:
    print(sentence)
print(vectorized.toarray())

for i, sentence in enumerate(test_sentences):
    label = "Anksiyete" if predictions[i] == 1 else "Normal"
    print(f"Cümle: {sentence}\n→ Tahmin: {label}\n")
