import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

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

        if "processed_tags" not in train.columns:
            raise KeyError("Column 'processed_tags' not found in the datasets.")

        # Combine all datasets to fit the TF-IDF vectorizer
        print("Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(train["processed_tags"].fillna(""))

        # Transform datasets
        print("Transforming datasets...")
        X_train = vectorizer.transform(train["processed_tags"].fillna(""))
        X_val = vectorizer.transform(val["processed_tags"].fillna(""))
        X_test = vectorizer.transform(test["processed_tags"].fillna(""))

        # Labels (e.g., 'Category' column)
        y_train = train["Category"]
        y_val = val["Category"]
        y_test = test["Category"]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save transformed datasets
        print("Saving TF-IDF matrices and labels...")
        pd.DataFrame(X_train.toarray()).to_csv(f"{output_dir}/X_train.csv", index=False)
        pd.DataFrame(X_val.toarray()).to_csv(f"{output_dir}/X_val.csv", index=False)
        pd.DataFrame(X_test.toarray()).to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        print(f"Processed datasets saved to {output_dir}.")

    except Exception as e:
        print(f"An error occurred during text representation: {e}")
        raise
