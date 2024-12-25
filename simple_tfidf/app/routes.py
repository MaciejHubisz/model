from flask import Blueprint, render_template, request
from simple_tfidf.app.models import predict_top3, predict_user_model, train_user_model
from simple_tfidf.app.utils import load_user_data, save_user_data

main = Blueprint('main', __name__)

countries = ["Afghanistan", "Iraq", "Pakistan", "India", "Syria", "Somalia", "Yemen", "Nigeria", "United States", "Colombia"]
weapons = ["Explosives", "Firearms", "Incendiary", "Melee", "Vehicle", "Unknown"]
decades = [str(decade) for decade in range(1970, 2020, 10)]

@main.route("/", methods=["GET", "POST"])
def index():
    print("dddduuuuupppppaaaaaa")
    selected_country = None
    selected_weapon = None
    selected_decade = None
    predictions = None
    user_predictions = None
    train_message = None
    user_data = load_user_data()

    if request.method == "POST":
        action = request.form.get("action")
        selected_country = request.form.get("country")
        selected_weapon = request.form.get("weapon")
        selected_decade = request.form.get("decade")

        if action == "predict":
            # Wylicz predykcje
            predictions = predict_top3(selected_country, selected_weapon, selected_decade)
            user_predictions = predict_user_model(selected_country, selected_weapon, selected_decade)

        elif action == "save":
            selected_option = request.form.get("selected_option")
            if selected_option:
                print("Wywoływanie save user data...")
                save_user_data(selected_country, selected_weapon, selected_decade, selected_option)
                print("Wywoływanie train_user_model...")
                train_message = train_user_model()
                print(f"Komunikat treningowy: {train_message}")
            else:
                print("Nie wybrano żadnej opcji. Model nie zostanie wytrenowany.")

    return render_template(
        "index.html",
        countries=countries,
        weapons=weapons,
        decades=decades,
        predictions=predictions,
        user_predictions=user_predictions,
        selected_country=selected_country,
        selected_weapon=selected_weapon,
        selected_decade=selected_decade,
        user_data=user_data.to_dict(orient="records"),
        train_message=train_message
    )
