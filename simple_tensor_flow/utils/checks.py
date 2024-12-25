def check_installed_libraries():
    """
    Checks if required libraries are installed and operational.
    """
    try:
        import numpy
        import pandas
        import matplotlib
        import tensorflow
        import keras
        import sklearn
        import nltk
        import spacy

        # Test a simple TensorFlow operation
        import tensorflow as tf
        tf.debugging.assert_equal(tf.constant(1) + tf.constant(1), tf.constant(2))

        # Test NLTK and spaCy
        nltk.download('stopwords', quiet=True)
        spacy.load("en_core_web_sm")

        print("✅ All libraries are successfully loaded.")
    except ImportError as e:
        print(f"❌ Library not found: {e.name}")
        raise
    except Exception as e:
        print(f"❌ An error occurred during the check: {str(e)}")
        raise
