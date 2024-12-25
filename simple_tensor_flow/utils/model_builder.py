from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def build_model(input_dim, num_classes):
    """
    Builds and compiles a neural network model.

    Parameters:
        input_dim (int): Dimension of the input data (TF-IDF features).
        num_classes (int): Number of output classes (categories).

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy if labels are integers
        metrics=['accuracy']
    )

    return model
