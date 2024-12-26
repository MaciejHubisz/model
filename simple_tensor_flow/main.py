import numpy as np
import pandas as pd

from simple_tensor_flow.utils.data_split import split_dataset
from simple_tensor_flow.utils.model_builder import build_model
from simple_tensor_flow.utils.model_training import train_model
from simple_tensor_flow.utils.text_representation import represent_text
from utils.kaggle_downloader import download_kaggle_dataset
from utils.checks import check_installed_libraries
from utils.data_analysis import analyze_dataset
from utils.preprocessing import preprocess_dataset
import os


def main():
    """
    Main entry point of the application.
    Handles library verification, dataset downloading, analysis, and preprocessing.
    """
    # 1. Sprawdź, czy wszystkie komponenty są zainstalowane
    print("Step 1: Verifying components...")
    check_installed_libraries()

    # 2. Pobierz dataset, jeśli go nie ma
    dataset_name = "andkonar/book-dataset-300k-scraped-polish"
    output_path = "datasets/"
    dataset_file = os.path.join(output_path, "book_data.csv")  # Dopasuj nazwę pliku do rzeczywistości

    if not os.path.exists(dataset_file):
        print("Step 2: Dataset not found. Downloading dataset...")
        # download_kaggle_dataset(dataset_name, output_path)
    else:
        print("Step 2: Dataset already exists. Skipping download.")

    # 3. Analizuj dataset
    print("Step 3: Analyzing dataset...")
    analyze_dataset(dataset_file)

    # 4. Przetwarzaj dataset
    preprocessed_file = os.path.join(output_path, "book_data_preprocessed.csv")
    if not os.path.exists(preprocessed_file):
        print("Step 4: Preprocessing dataset...")
        preprocess_dataset(dataset_file, preprocessed_file)
    else:
        print("Step 4: Preprocessed dataset already exists. Skipping preprocessing.")

    # 5. Podziel dataset
    print("Step 5: Splitting dataset...")
    split_dir = os.path.join(output_path, "splits")
    split_dataset(preprocessed_file, split_dir)

    # 6. Reprezentacja tekstu
    print("Step 6: Representing text as TF-IDF...")
    tfidf_dir = os.path.join(output_path, "tfidf")
    represent_text(
        os.path.join(split_dir, "train.csv"),
        os.path.join(split_dir, "validation.csv"),
        os.path.join(split_dir, "test.csv"),
        tfidf_dir
    )

    # 7. Budowa modelu
    print("Step 7: Building the neural network model...")
    X_train = pd.read_csv(f"{tfidf_dir}/X_train.csv").values
    y_train = pd.read_csv(f"{tfidf_dir}/y_train.csv").values.flatten()

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = build_model(input_dim, num_classes)
    model.summary()

    # # Zapis modelu do pliku
    model_path = os.path.join(output_path, "model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 8. Trenowanie modelu
    print("Step 8: Training the model...")
    trained_model_path = os.path.join(output_path, "trained_model.keras")
    train_model(
        model_path,
        tfidf_dir,
        tfidf_dir,
        trained_model_path,
        epochs=20,
        batch_size=32
    )


if __name__ == "__main__":
    main()
