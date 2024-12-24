import os
import pickle
from flask import Flask, request, render_template

# Ścieżka do folderu z plikami modelu i vectorizera
DATA_FOLDER = "data"
MODEL_PATH = os.path.join(DATA_FOLDER, "model.pkl")
VECTORIZER_PATH = os.path.join(DATA_FOLDER, "vectorizer.pkl")

# Wczytaj model i vectorizer
with open(MODEL_PATH, "rb") as f_model:
    model = pickle.load(f_model)

with open(VECTORIZER_PATH, "rb") as f_vec:
    vectorizer = pickle.load(f_vec)

# Flask app
app = Flask(__name__)

# Lista opcji do dropdownów
countries = ["Afghanistan", "Iraq", "Pakistan", "India", "Syria", "Somalia", "Yemen", "Nigeria", "United States",
             "Colombia"]
weapons = ["Explosives", "Firearms", "Incendiary", "Melee", "Vehicle", "Unknown"]
decades = [str(decade) for decade in range(1970, 2020, 10)]  # Dekady od 1970 do 2010


def predict_top3(country, weapon, decade):
    """
    Funkcja przewidująca top 3 rodzaje ataków z prawdopodobieństwami.
    """
    # Tworzenie tekstu wejściowego
    user_input = f"{country} {decade} {weapon}"

    # Wektorowanie danych
    input_vectorized = vectorizer.transform([user_input])

    # Przewidywanie prawdopodobieństw
    probabilities = model.predict_proba(input_vectorized)[0]
    classes = model.classes_

    # Łączenie prawdopodobieństw z klasami
    predictions = list(zip(classes, probabilities))

    # Sortowanie i wybranie top 3
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
    return [(label, round(prob * 100, 2)) for label, prob in predictions_sorted]


@app.route("/", methods=["GET", "POST"])
def index():
    # Domyślne wartości formularza
    selected_country = None
    selected_weapon = None
    selected_decade = None
    predictions = None

    if request.method == "POST":
        # Pobierz dane z formularza
        selected_country = request.form.get("country")
        selected_weapon = request.form.get("weapon")
        selected_decade = request.form.get("decade")

        # Przewiduj top 3
        predictions = predict_top3(selected_country, selected_weapon, selected_decade)

    return render_template(
        "index.html",
        countries=countries,
        weapons=weapons,
        decades=decades,
        predictions=predictions,
        selected_country=selected_country,
        selected_weapon=selected_weapon,
        selected_decade=selected_decade
    )


if __name__ == "__main__":
    app.run(debug=True)
