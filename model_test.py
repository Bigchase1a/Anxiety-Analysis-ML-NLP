import pickle
from text_preprocessing import preprocess_text

with open("anxiety_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vec_file:
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

for i, sentence in enumerate(test_sentences):
    label = "Anksiyete" if predictions[i] == 1 else "Normal"
    print(f"Cümle: {sentence}\n→ Tahmin: {label}\n")
