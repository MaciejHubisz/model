import pandas as pd
import spacy
import re


def preprocess_tags(tags, nlp):
    """
    Cleans and preprocesses a single 'Tags' entry.

    Parameters:
        tags (str): Input tags to preprocess.
        nlp: SpaCy language model.

    Returns:
        str: Preprocessed tags, joined by commas.
    """
    if pd.isna(tags):
        return ""

    # Split tags into a list (by comma) and normalize spacing
    tags_list = [tag.strip() for tag in tags.split(",")]

    # Process each tag: lowercase, remove special characters, lemmatize
    processed_tags = []
    for tag in tags_list:
        tag = tag.lower()  # Lowercase
        tag = re.sub(r"[^a-ząćęłńóśźż\s]", "", tag)  # Remove special characters
        doc = nlp(tag)  # Process using SpaCy
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        if lemmatized:  # Skip empty results
            processed_tags.append(lemmatized)

    # Remove duplicates and join back into a single string
    unique_tags = list(dict.fromkeys(processed_tags))  # Remove duplicates
    return ",".join(unique_tags)


def preprocess_dataset(input_path, output_path):
    """
    Preprocesses the dataset by cleaning and lemmatizing the 'Tags' column.

    Parameters:
        input_path (str): Path to the input dataset file (CSV).
        output_path (str): Path to save the preprocessed dataset.
    """
    try:
        print(f"Loading dataset from {input_path}...")
        data = pd.read_csv(input_path)

        # Ensure the necessary column exists
        if "Tags" not in data.columns:
            raise KeyError("Column 'Tags' not found in dataset.")

        print("Loading SpaCy model...")
        nlp = spacy.load("pl_core_news_sm")

        print("Preprocessing 'Tags' column...")
        data["processed_tags"] = data["Tags"].apply(lambda x: preprocess_tags(str(x), nlp))

        print(f"Saving preprocessed dataset to {output_path}...")
        data.to_csv(output_path, index=False)

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise
