import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

def convert_labels_to_numeric(data, label_column):
    """
    Converts text labels to numeric values.

    Parameters:
        data (pd.DataFrame): The dataset containing the labels.
        label_column (str): The name of the column with labels.

    Returns:
        data (pd.DataFrame): The updated dataset with numeric labels.
        label_map (dict): A dictionary mapping labels to numeric values.
    """
    label_map = {label: idx for idx, label in enumerate(data[label_column].unique())}
    data[label_column] = data[label_column].map(label_map)
    return data, label_map

def represent_text(train_path, val_path, test_path, output_dir):
    """
    Converts text data into numerical representation using TF-IDF.

    Parameters:
        train_path (str): Path to the training dataset (CSV).
        val_path (str): Path to the validation dataset (CSV).
        test_path (str): Path to the test dataset (CSV).
        output_dir (str): Directory to save the processed datasets.

    Returns:
        None: Saves the TF-IDF matrices and labels as files.
    """
    try:
        print("Loading datasets...")
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        if "processed_tags" not in train.columns or "Category" not in train.columns:
            raise KeyError("Required columns 'processed_tags' or 'Category' not found in the datasets.")

        # Convert labels to numeric values
        print("Converting labels to numeric values...")
        train, label_map = convert_labels_to_numeric(train, "Category")
        val["Category"] = val["Category"].map(label_map)  # Ensure consistent mapping
        test["Category"] = test["Category"].map(label_map)  # Ensure consistent mapping

        # Save label map for later use
        with open(f"{output_dir}/label_map.json", "w") as f:
            json.dump(label_map, f)

        # Combine all datasets to fit the TF-IDF vectorizer
        print("Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(train["processed_tags"].fillna(""))

        # Transform datasets
        print("Transforming datasets...")
        X_train = vectorizer.transform(train["processed_tags"].fillna(""))
        X_val = vectorizer.transform(val["processed_tags"].fillna(""))
        X_test = vectorizer.transform(test["processed_tags"].fillna(""))

        # Save transformed datasets
        print("Saving TF-IDF matrices and labels...")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(X_train.toarray()).to_csv(f"{output_dir}/X_train.csv", index=False)
        pd.DataFrame(X_val.toarray()).to_csv(f"{output_dir}/X_val.csv", index=False)
        pd.DataFrame(X_test.toarray()).to_csv(f"{output_dir}/X_test.csv", index=False)
        train["Category"].to_csv(f"{output_dir}/y_train.csv", index=False)
        val["Category"].to_csv(f"{output_dir}/y_val.csv", index=False)
        test["Category"].to_csv(f"{output_dir}/y_test.csv", index=False)

        print(f"Processed datasets saved to {output_dir}.")

    except Exception as e:
        print(f"An error occurred during text representation: {e}")
        raise
