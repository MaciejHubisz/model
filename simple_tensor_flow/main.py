from utils.kaggle_downloader import download_kaggle_dataset
from utils.checks import check_installed_libraries
from utils.data_analysis import analyze_dataset
import os

def main():
    """
    Main entry point of the application.
    Handles library verification, dataset downloading, and analysis.
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
        download_kaggle_dataset(dataset_name, output_path)
    else:
        print("Step 2: Dataset already exists. Skipping download.")

    # 3. Analizuj dataset
    print("Step 3: Analyzing dataset...")
    analyze_dataset(dataset_file)

if __name__ == "__main__":
    main()