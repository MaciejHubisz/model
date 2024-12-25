import os

import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_path, output_dir, test_size=0.15, validation_size=0.15, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        input_path (str): Path to the input dataset (CSV).
        output_dir (str): Directory to save the split datasets.
        test_size (float): Proportion of the dataset to include in the test set.
        validation_size (float): Proportion of the dataset to include in the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None: Saves the split datasets as CSV files.
    """
    try:
        print(f"Loading dataset from {input_path}...")
        data = pd.read_csv(input_path)

        # Split into train + validation and test sets
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        # Calculate validation size relative to the train+validation set
        val_relative_size = validation_size / (1 - test_size)

        # Split train+validation into training and validation sets
        train, val = train_test_split(
            train_val, test_size=val_relative_size, random_state=random_state
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the splits
        train.to_csv(f"{output_dir}/train.csv", index=False)
        val.to_csv(f"{output_dir}/validation.csv", index=False)
        test.to_csv(f"{output_dir}/test.csv", index=False)

        print(f"Datasets saved to {output_dir}:")
        print(f"- Training set: {len(train)} rows")
        print(f"- Validation set: {len(val)} rows")
        print(f"- Test set: {len(test)} rows")

    except Exception as e:
        print(f"An error occurred during dataset splitting: {e}")
        raise
