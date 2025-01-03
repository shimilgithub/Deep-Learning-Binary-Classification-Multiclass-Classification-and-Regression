"""implements regression using the Boston Housing dataset with TensorFlow."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing

class BostonHousing:
    """Regression on the Boston Housing dataset"""

    def __init__(self):
        """Initialize the BostonHousing class."""
        self.model = None

    def prepare_data(self):
        """Load and preprocess the Boston Housing dataset."""
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

        # Normalize the data
        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)
        self.x_train = (x_train - self.mean) / self.std
        self.x_test = (x_test - self.mean) / self.std
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        """Build a regression model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),  
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(64, activation='relu'),   
            tf.keras.layers.Dense(1)  
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, epochs=40, batch_size=16):
        """Train the model with the training data."""
        self.history = self.model.fit(
            self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.2
        )

    def plot_loss(self):
        """Plot training and validation loss."""
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'b-.', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_mae(self):
        """Plots the training and validation Mean Absolute Error (MAE)."""
        mae = self.history.history['mae']
        val_mae = self.history.history['val_mae']
        epochs = range(1, len(mae) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mae, 'b', label='Training MAE')
        plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        plt.title('Training and Validation Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    def calculate_mse_rmse(self):
        """Calculate and return the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)."""
        predictions = self.model.predict(self.x_test)
        mse = np.mean((predictions.flatten() - self.y_test) ** 2)
        rmse = np.sqrt(mse)
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

    def evaluate(self):
        """Evaluate the model"""
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)

