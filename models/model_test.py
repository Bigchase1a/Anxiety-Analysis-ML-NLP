import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from helpers.text_preprocessing import preprocess_text

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
model_path = os.path.join(base_path, "anxiety_model_20250510_0215.pkl")
vec_path = os.path.join(base_path, "tfidf_vectorizer_20250510_0215.pkl")

with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open(vec_path, "rb") as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

test_sentences = [
    "I hate everything, I don't want to live anymore.",
    "I feel happy and excited for the day ahead.",
    "Life feels like a burden I can’t carry anymore.",
    "I enjoyed a lovely walk in the park today.",
    "I just want to disappear and never come back.",
    "I’m looking forward to spending time with my friends.",
    "Nothing I do seems to matter; I feel so empty.",
    "I’m grateful for the little things that make me smile.",
    "Even surrounded by people, I feel completely alone.",
    "Today was a productive and fulfilling day."
]

preprocessed = [preprocess_text(text) for text in test_sentences]
vectorized = loaded_vectorizer.transform(preprocessed)
predictions = loaded_model.predict(vectorized)
probas = loaded_model.predict_proba(vectorized)

print("Test Modeli: ", os.path.basename(model_path).replace(".pkl", ""))
def interpret_score(score):
    return "Anksiyete" if score >= 0.5 else "Normal"

for i, sentence in enumerate(test_sentences):
    anxiety_score = probas[i][1]
    label = interpret_score(anxiety_score)
    print(f"{sentence}\n→{label} ({anxiety_score * 100:.2f}%)")