import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score
from helpers.text_preprocessing import preprocess_text

# Veriyi Yükle
data = pd.read_csv("data/mental_health_corpus.csv")
data.dropna(inplace=True)
print("Model Eğitim Başlangıcı: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Etiket ve Metni Ayır
texts = data['text']
labels = data['label']

# Etiket Dağılımı Grafiği
label_counts = labels.value_counts()
label_counts.index = label_counts.index.map({0: 'Normal', 1: 'Anksiyete'})

plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('"Anksiyete" ve "Normal" Metinlerin Dağılımı')
plt.xlabel('Etiketler')
plt.ylabel('Sayı')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Ön İşleme
processed_texts = texts.apply(preprocess_text)
data["processed"] = processed_texts

# Eğitim/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.2, random_state=32, stratify=labels)

# TF-IDF Vektörizasyonu
tfidf = TfidfVectorizer(max_features=20000, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Eğitimi
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Cross Validation
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("5-Fold Cross Validation Ortalama Doğruluk: {:.4f}".format(cv_scores.mean()))

# Tahmin
y_pred = model.predict(X_test_tfidf)
y_probs = model.predict_proba(X_test_tfidf)[:, 1]

# Değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_probs))
print(classification_report(y_test, y_pred, target_names=["Normal", "Anksiyete"]))
print("Model Eğitim Sonu: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anksiyete"], yticklabels=["Normal", "Anksiyete"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# TF-IDF Özellik Önem Skoru
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]
top_features = sorted(zip(coefficients, feature_names), reverse=True)[:20]

top_coefs = pd.DataFrame(top_features, columns=["Coef", "Feature"])
plt.figure(figsize=(10, 5))
sns.barplot(data=top_coefs, x="Coef", y="Feature", palette="viridis")
plt.title("Anksiyete Sınıfı İçin En Etkili 20 Kelime")
plt.tight_layout()
plt.show()

# Metin Analizi: Token istatistikleri
all_tokens = [word for text in processed_texts for word in text.split()]
unique_words = set(all_tokens)
print(f"Benzersiz kelime sayısı: {len(unique_words)}")

# Ortalama Cümle Uzunluğu
data["text_len"] = data["processed"].apply(lambda x: len(x.split()))
sns.boxplot(data=data, x="label", y="text_len")
plt.title("Cümle Uzunlukları - Sınıf Bazında")
plt.xlabel("Label (0: Normal, 1: Anksiyete)")
plt.ylabel("Kelime Sayısı")
plt.tight_layout()
plt.show()

# WordCloud - Normal
text_0 = " ".join(data[data['label'] == 0]['processed'])
wc_0 = WordCloud(width=800, height=400, background_color='white').generate(text_0)
plt.figure(figsize=(10, 5))
plt.imshow(wc_0, interpolation='bilinear')
plt.axis('off')
plt.title("Normal Metinlerde En Sık Geçen Kelimeler")
plt.tight_layout()
plt.show()

# WordCloud - Anksiyete
text_1 = " ".join(data[data['label'] == 1]['processed'])
wc_1 = WordCloud(width=800, height=400, background_color='white').generate(text_1)
plt.figure(figsize=(10, 5))
plt.imshow(wc_1, interpolation='bilinear')
plt.axis('off')
plt.title("Anksiyete Metinlerinde En Sık Geçen Kelimeler")
plt.tight_layout()
plt.show()

# Model ve Vektörizeri Kaydet
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs("models", exist_ok=True)
with open(f"models/anxiety_model_{timestamp}.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open(f"models/tfidf_vectorizer_{timestamp}.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)
