import os
import pandas as pd

DATA_FOLDER = "data"
USER_DATA_PATH = os.path.join(DATA_FOLDER, "user_data.csv")

def load_user_data():
    """
    Wczytuje dane użytkownika z pliku CSV.
    """
    if os.path.exists(USER_DATA_PATH):
        return pd.read_csv(USER_DATA_PATH)
    return pd.DataFrame(columns=["country", "weapon", "decade", "selected_option"])

def save_user_data(country, weapon, decade, selected_option):
    """
    Zapisuje dane użytkownika do pliku CSV.
    """
    user_data = load_user_data()
    new_entry = pd.DataFrame([{
        "country": country,
        "weapon": weapon,
        "decade": decade,
        "selected_option": selected_option
    }])
    user_data = pd.concat([user_data, new_entry], ignore_index=True)
    user_data.to_csv(USER_DATA_PATH, index=False)
