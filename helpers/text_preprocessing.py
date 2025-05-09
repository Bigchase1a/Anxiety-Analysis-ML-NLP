import re
import nltk
import spacy
import contractions
from nltk.corpus import stopwords
import importlib.resources
from symspellpy import SymSpell

nltk.download('stopwords')
nltk.download('wordnet')
spacy.require_gpu()
nlp = spacy.load("en_core_web_md")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

# Yazım düzeltme
def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# Stop words temizleme
def remove_stopwords(text, custom_stopwords=None):
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# POS etiketleme ile lemmatizasyon
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized = []
    for token in doc:
        if token.pos_ not in ['PROPN', 'DET'] and token.is_alpha:
            lemmatized.append(token.lemma_ if token.lemma_ else token.text)
        else:
            lemmatized.append(token.text)
    return ' '.join(lemmatized)

# Metin Temizleme
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    text = re.sub(r"@\S+", "USER", text)
    text = re.sub(r"\d+", "NUMBER", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Ön işleme
def preprocess_text(text, custom_stopwords=None):
    text = clean_text(text)
    text = correct_spelling(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text, custom_stopwords)
    return text
