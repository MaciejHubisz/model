import pandas as pd
import os


def analyze_dataset(dataset_path):
    """
    Loads and analyzes the dataset.

    Parameters:
        dataset_path (str): Path to the dataset file (CSV or other format).

    Returns:
        None: Prints basic statistics and structure of the dataset.
    """
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")

        # Load the dataset
        print(f"Loading dataset from: {dataset_path}")
        data = pd.read_csv(dataset_path)

        # Print basic info
        print("\nDataset Info:")
        print(data.info())

        # Print first few rows
        print("\nFirst 5 rows:")
        print(data.head())

        # Print basic statistics
        print("\nBasic Statistics:")
        print(data.describe(include="all"))

        # Check for missing values
        print("\nMissing Values:")
        print(data.isnull().sum())

    except Exception as e:
        print(f"An error occurred during dataset analysis: {e}")
        raise
