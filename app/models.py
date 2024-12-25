import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_FOLDER = "data"
USER_DATA_PATH = os.path.join(DATA_FOLDER, "user_data.csv")
USER_MODEL_PATH = os.path.join(DATA_FOLDER, "user_model.pkl")
USER_VECTORIZER_PATH = os.path.join("data", "user_vectorizer.pkl")

# Wczytaj główny model i vectorizer
with open(os.path.join(DATA_FOLDER, "model.pkl"), "rb") as f_model:
    model = pickle.load(f_model)

with open(os.path.join(DATA_FOLDER, "vectorizer.pkl"), "rb") as f_vec:
    vectorizer = pickle.load(f_vec)

def predict_top3(country, weapon, decade):
    """
    Funkcja przewidująca top 3 rodzaje ataków z prawdopodobieństwami.
    """
    user_input = f"{country} {decade} {weapon}"
    input_vectorized = vectorizer.transform([user_input])
    probabilities = model.predict_proba(input_vectorized)[0]
    classes = model.classes_
    predictions = list(zip(classes, probabilities))
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
    return [(label, round(prob * 100, 2)) for label, prob in predictions_sorted]

def train_user_model():
    """
    Funkcja trenująca nowy model na podstawie danych użytkownika.
    """
    return "dupa"
    user_data = pd.read_csv(USER_DATA_PATH)
    if user_data.empty:
        return "Brak danych użytkownika do trenowania modelu."

    user_data["text_data"] = user_data["country"] + " " + user_data["decade"].astype(str) + " " + user_data["weapon"]
    X = user_data["text_data"]
    y = user_data["selected_option"]

    user_vectorizer = TfidfVectorizer()
    X_tfidf = user_vectorizer.fit_transform(X)

    user_model = LogisticRegression(max_iter=1000)
    user_model.fit(X_tfidf, y)

    with open(USER_MODEL_PATH, "wb") as f_user_model:
        pickle.dump(user_model, f_user_model)

    with open(os.path.join(DATA_FOLDER, "user_vectorizer.pkl"), "wb") as f_user_vec:
        pickle.dump(user_vectorizer, f_user_vec)

    return "Nowy model użytkownika został wytrenowany i zapisany."

def predict_user_model(country, weapon, decade):
    """
    Przewiduje wyniki na podstawie modelu użytkownika.
    """
    if not os.path.exists(USER_MODEL_PATH) or not os.path.exists(USER_VECTORIZER_PATH):
        return {}

    with open(USER_MODEL_PATH, "rb") as f_model:
        user_model = pickle.load(f_model)

    with open(USER_VECTORIZER_PATH, "rb") as f_vec:
        user_vectorizer = pickle.load(f_vec)

    user_input = f"{country} {decade} {weapon}"
    input_vectorized = user_vectorizer.transform([user_input])
    probabilities = user_model.predict_proba(input_vectorized)[0]
    classes = user_model.classes_
    return {label: round(prob * 100, 2) for label, prob in zip(classes, probabilities)}
