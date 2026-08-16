import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

# ----------------------------------------------------------------------
# 1. CARGA DE DATOS
# ----------------------------------------------------------------------

# Debes usar un archivo con dos columnas: "label" y "text"
# label = "spam" o "ham"
# Ejemplo:
# df = pd.read_csv("spam.csv", encoding='latin-1')[['v1','v2']]
# df.columns = ['label','text']

df = pd.read_csv("dataset_spam.csv")   # <--- ajusta el nombre

# ----------------------------------------------------------------------
# 2. LIMPIEZA DEL TEXTO
# ----------------------------------------------------------------------

try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    # Si falla la descarga de stopwords, usar un conjunto vacío
    stop_words = set()

# Inicializar el stemmer para reducir palabras a su raíz
stemmer = PorterStemmer()

def limpiar_texto(text):
    text = text.lower()  # minúsculas
    text = re.sub(r"http\S+", "", text)  # elimina URLs
    text = re.sub(r"\d+", "", text)  # elimina números
    text = text.translate(str.maketrans("", "", string.punctuation))  # quita signos
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    # Aplicar stemming a cada token
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

df['clean'] = df['text'].astype(str).apply(limpiar_texto)

# ----------------------------------------------------------------------
# 3. TF-IDF
# ----------------------------------------------------------------------

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean'])
y = df['label']

# ----------------------------------------------------------------------
# 4. DIVISIÓN TRAIN / TEST
# ----------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------------------------------------------
# 5. ENTRENAMIENTO DE LOS 3 MODELOS
# ----------------------------------------------------------------------

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

# kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# SVM
svm = LinearSVC(max_iter=1000)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

# ----------------------------------------------------------------------
# 6. MÉTRICAS
# ----------------------------------------------------------------------

print("\n================= NAIVE BAYES =================")
print(classification_report(y_test, pred_nb))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_nb))

print("\n================= kNN =================")
print(classification_report(y_test, pred_knn))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_knn))

print("\n================= SVM =================")
print(classification_report(y_test, pred_svm))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, pred_svm))

# ----------------------------------------------------------------------
# 7. RESUMEN COMPARATIVO
# ----------------------------------------------------------------------

from sklearn.metrics import accuracy_score

print("\n" + "="*60)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*60)

resultados = {
    'Naive Bayes': accuracy_score(y_test, pred_nb),
    'kNN': accuracy_score(y_test, pred_knn),
    'SVM': accuracy_score(y_test, pred_svm)
}

for modelo, accuracy in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    print(f"{modelo:15s}: {accuracy*100:.2f}% accuracy")

mejor_modelo = max(resultados, key=resultados.get)
print(f"\n🏆 Mejor modelo: {mejor_modelo} con {resultados[mejor_modelo]*100:.2f}% accuracy")

print("\n" + "="*60)
print("INFORMACIÓN DEL DATASET")
print("="*60)
print(f"Total de mensajes: {len(df)}")
print(f"Mensajes de entrenamiento: {len(X_train)}")
print(f"Mensajes de prueba: {len(X_test)}")
print(f"\nDistribución:")
print(df['label'].value_counts())
print(f"\nPorcentajes:")
print(df['label'].value_counts(normalize=True) * 100)
