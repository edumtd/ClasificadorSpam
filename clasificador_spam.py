"""
Clasificador de spam: Naive Bayes vs kNN vs SVM sobre TF-IDF.

Uso:
    python clasificador_spam.py                       # usa dataset_spam.csv
    python clasificador_spam.py --datos SMSSpamCollection
    python clasificador_spam.py --cv 5                # agrega validación cruzada
    python clasificador_spam.py --con-duplicados      # no elimina mensajes repetidos
    python clasificador_spam.py --guardar modelo.joblib
"""

import argparse
import re
import string
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEMILLA = 42

# ----------------------------------------------------------------------
# 1. CARGA DE DATOS
# ----------------------------------------------------------------------

def cargar_datos(ruta):
    """Carga un CSV con columnas label,text o el archivo SMSSpamCollection (TSV)."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
    else:
        df = pd.read_csv(ruta, sep="\t", header=None, names=["label", "text"],
                         quoting=3, encoding="utf-8")
    if not {"label", "text"} <= set(df.columns):
        raise ValueError(f"{ruta} debe tener columnas 'label' y 'text'")
    df = df[["label", "text"]].dropna()
    if df.empty:
        raise ValueError(f"{ruta}: no hay filas con 'label' y 'text' válidos "
                         "(¿el archivo está mal formado?)")
    if not set(df["label"].unique()) <= {"ham", "spam"}:
        raise ValueError(f"{ruta}: 'label' debe contener solo 'ham'/'spam' "
                         f"(se encontró {df['label'].unique()[:5]})")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. LIMPIEZA DEL TEXTO
# ----------------------------------------------------------------------

def _cargar_stopwords():
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        import nltk
        try:
            nltk.download("stopwords", quiet=True)
            return frozenset(stopwords.words("english"))
        except Exception:
            print("⚠ No se pudieron descargar las stopwords; se continúa sin ellas.")
            return frozenset()


STOP_WORDS = _cargar_stopwords()
_STEMMER = PorterStemmer()
# Se compilan una sola vez (antes se recompilaban en cada mensaje)
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_NUM = re.compile(r"\d+")
_TABLA_PUNT = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=None)
def _stem(palabra):
    # El vocabulario se repite mucho: cachear evita recalcular la raíz
    return _STEMMER.stem(palabra)


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_NUM.sub(" ", texto)
    texto = texto.translate(_TABLA_PUNT)
    return " ".join(_stem(t) for t in texto.split() if t not in STOP_WORDS)


# ----------------------------------------------------------------------
# 3. MODELOS (Pipeline = TF-IDF + clasificador)
# ----------------------------------------------------------------------
# Con Pipeline, TF-IDF se ajusta SOLO con los datos de entrenamiento.
# Antes se ajustaba con todo el dataset, lo que filtraba información del
# conjunto de prueba (vocabulario e IDF) hacia el entrenamiento.

def crear_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2),   # palabras y pares de palabras ("free entri", "call now")
        sublinear_tf=True,    # 1 + log(tf): reduce el peso de palabras repetidas
        min_df=2,             # ignora términos que aparecen en un solo mensaje
        dtype=np.float32,      # la mitad de memoria que float64
    )


def crear_modelos():
    return {
        "Naive Bayes": Pipeline([("tfidf", crear_tfidf()),
                                 ("clf", MultinomialNB(alpha=0.1))]),
        # Coseno es la distancia adecuada para vectores TF-IDF
        "kNN": Pipeline([("tfidf", crear_tfidf()),
                         ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                                      weights="distance", n_jobs=-1))]),
        "SVM": Pipeline([("tfidf", crear_tfidf()),
                         ("clf", LinearSVC(C=1.0, random_state=SEMILLA))]),
    }


# ----------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Clasificador de spam (NB, kNN, SVM)")
    p.add_argument("--datos", default="dataset_spam.csv")
    p.add_argument("--test", type=float, default=0.3, help="proporción de prueba")
    p.add_argument("--cv", type=int, default=0, help="k de validación cruzada (0 = no)")
    p.add_argument("--con-duplicados", action="store_true",
                   help="conserva mensajes repetidos (pueden caer en train y test a la vez)")
    p.add_argument("--guardar", metavar="ARCHIVO", help="guarda el mejor modelo con joblib")
    args = p.parse_args()

    df = cargar_datos(args.datos)
    total_original = len(df)

    t0 = time.perf_counter()
    df["clean"] = df["text"].map(limpiar_texto)
    df = df[df["clean"].str.strip() != ""]
    print(f"Limpieza: {time.perf_counter() - t0:.2f} s")

    if not args.con_duplicados:
        df = df.drop_duplicates(subset="clean")
    df = df.reset_index(drop=True)

    X, y = df["clean"], df["label"]

    # stratify mantiene la misma proporción ham/spam en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test, random_state=SEMILLA, stratify=y)

    resultados = {}
    modelos = crear_modelos()
    for nombre, modelo in modelos.items():
        t0 = time.perf_counter()
        modelo.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = modelo.predict(X_test)
        t_pred = time.perf_counter() - t0

        resultados[nombre] = {
            "accuracy": accuracy_score(y_test, pred),
            "f1_spam": f1_score(y_test, pred, pos_label="spam"),
        }
        print(f"\n================= {nombre.upper()} =================")
        print(f"Entrenamiento: {t_fit:.3f} s | Predicción: {t_pred:.3f} s")
        print(classification_report(y_test, pred, digits=4))
        print("Matriz de Confusión (filas = real, columnas = predicho) [ham, spam]:")
        print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))

    # ------------------------------------------------------------------
    # RESUMEN COMPARATIVO
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO DE MODELOS (conjunto de prueba)")
    print("=" * 60)
    # Con clases desbalanceadas el F1 de spam es mejor criterio que el accuracy
    orden = sorted(resultados.items(), key=lambda kv: kv[1]["f1_spam"], reverse=True)
    for nombre, r in orden:
        print(f"{nombre:12s}: accuracy {r['accuracy'] * 100:6.2f}% | F1 spam {r['f1_spam']:.4f}")
    mejor = orden[0][0]
    print(f"\n🏆 Mejor modelo (por F1 de spam): {mejor}")

    if args.cv > 1:
        print("\n" + "=" * 60)
        print(f"VALIDACIÓN CRUZADA ESTRATIFICADA ({args.cv} particiones)")
        print("=" * 60)
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEMILLA)
        for nombre, modelo in crear_modelos().items():
            cv = cross_validate(modelo, X, y, cv=skf, n_jobs=-1,
                                scoring={"acc": "accuracy",
                                         "f1": "f1_macro"})
            print(f"{nombre:12s}: accuracy {cv['test_acc'].mean() * 100:6.2f}% "
                  f"± {cv['test_acc'].std() * 100:.2f} | "
                  f"F1 macro {cv['test_f1'].mean():.4f} ± {cv['test_f1'].std():.4f}")

    if args.guardar:
        # Se reentrena con todos los datos antes de guardar
        final = crear_modelos()[mejor].fit(X, y)
        joblib.dump(final, args.guardar)
        print(f"\nModelo '{mejor}' guardado en {args.guardar}. Para usarlo:")
        print("  m = joblib.load(ruta); m.predict([limpiar_texto('Win a free prize!')])")

    print("\n" + "=" * 60)
    print("INFORMACIÓN DEL DATASET")
    print("=" * 60)
    print(f"Mensajes en el archivo: {total_original}")
    print(f"Mensajes usados: {len(df)} "
          f"({total_original - len(df)} eliminados por vacíos o duplicados)")
    print(f"Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")
    conteo = y.value_counts()
    for etiqueta, n in conteo.items():
        print(f"  {etiqueta}: {n} ({n / len(y) * 100:.1f}%)")


if __name__ == "__main__":
    main()
