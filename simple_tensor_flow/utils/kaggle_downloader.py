import os
import json
import subprocess


def download_kaggle_dataset(dataset_name, output_path):
    """
    Downloads a dataset from Kaggle using the Kaggle API and a JSON token.

    Parameters:
        dataset_name (str): The Kaggle dataset identifier (e.g., 'username/dataset-name').
        output_path (str): The local path where the dataset will be saved.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Check for Kaggle token
        kaggle_token_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_token_path):
            raise FileNotFoundError("Kaggle token not found. Please place 'kaggle.json' in ~/.kaggle/")

        # Ensure proper permissions for the Kaggle token
        os.chmod(kaggle_token_path, 0o600)

        # Run Kaggle API command to download the dataset
        print(f"Downloading dataset: {dataset_name}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", output_path, "--unzip"],
            check=True
        )
        print("Dataset downloaded and unzipped successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
