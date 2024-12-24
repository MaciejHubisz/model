import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Folder na dataset i zapis modelu
DATASET_FOLDER = "datasets"
DATA_FOLDER = "data"
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Ścieżki do pliku datasetu i zapisanych modeli
DATASET_PATH = os.path.join(DATASET_FOLDER, "globalterrorismdb_0718dist.csv")
MODEL_PATH = os.path.join(DATA_FOLDER, "model.pkl")
VECTORIZER_PATH = os.path.join(DATA_FOLDER, "vectorizer.pkl")

# Sprawdź, czy plik datasetu istnieje
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Plik datasetu {DATASET_PATH} nie istnieje. Upewnij się, że pobrałeś dane.")

# 1. Wczytaj dane
df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")

# 2. Filtruj i wybierz istotne kolumny
df = df[["country_txt", "iyear", "weaptype1_txt", "attacktype1_txt"]]
df = df.dropna()  # Usuń brakujące dane

# 3. Stwórz kolumnę z dekadą
df["decade"] = (df["iyear"] // 10) * 10

# 4. Stwórz tekstową kolumnę wejściową
df["text_data"] = df["country_txt"] + " " + df["decade"].astype(str) + " " + df["weaptype1_txt"]

# 5. Ustaw dane wejściowe i etykiety
X_raw = df["text_data"]
y = df["attacktype1_txt"]

# 6. Podziel na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# 7. Wektoruj dane wejściowe
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# 8. Wytrenuj model klasyfikacji
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 9. Zapisz model i vectorizer do plików
with open(MODEL_PATH, "wb") as f_model:
    pickle.dump(model, f_model)

with open(VECTORIZER_PATH, "wb") as f_vec:
    pickle.dump(vectorizer, f_vec)

# 10. Wyświetl dokładność modelu na zbiorze testowym
X_test_tfidf = vectorizer.transform(X_test)
accuracy = model.score(X_test_tfidf, y_test)
print(f"Dokładność modelu: {accuracy:.2%}")
