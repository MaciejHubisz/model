import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

def train_model(model_path, train_data_path, val_data_path, output_model_path, epochs=20, batch_size=32):
    """
    Trains a saved model on the provided data and saves the trained model.

    Parameters:
        model_path (str): Path to the saved model file.
        train_data_path (str): Path to the training data file.
        val_data_path (str): Path to the validation data file.
        output_model_path (str): Path to save the trained model.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.

    Returns:
        None
    """
    try:
        # Load the model without the optimizer's state
        print("Loading model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Load training and validation data
        print("Loading training and validation data...")
        X_train = pd.read_csv(f"{train_data_path}/X_train.csv").values.astype(np.float32)
        y_train = pd.read_csv(f"{train_data_path}/y_train.csv").values.flatten().astype(np.int32)
        X_val = pd.read_csv(f"{val_data_path}/X_val.csv").values.astype(np.float32)
        y_val = pd.read_csv(f"{val_data_path}/y_val.csv").values.flatten().astype(np.int32)

        # Define callbacks
        checkpoint = ModelCheckpoint(output_model_path, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )

        # Save the final model
        model.save(output_model_path)
        print(f"Model trained and saved to {output_model_path}")

        # Plot training history
        print("Plotting training history...")
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise
