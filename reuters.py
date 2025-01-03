"""This script implements multiclass classification using the Reuters dataset with TensorFlow"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

class Reuters:
    """Multiclass classification on the Reuters dataset using a dense neural network."""

    def __init__(self, num_words=10000):
        """Initialize the Reuters class"""
        self.num_words = num_words
        self.model = None

    def prepare_data(self):
        """Load and preprocess dataset."""
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=self.num_words)

        self.x_train = self.vectorize_sequences(x_train)
        self.x_test = self.vectorize_sequences(x_test)
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

    def vectorize_sequences(self, sequences):
        """Convert sequences to binary matrix form."""
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        """Build a multiclass classification model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(46, activation='softmax')
        ])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=20, batch_size=512):
        """Train the model with the training data."""
        x_val = self.x_train[:1000]
        y_val = self.y_train[:1000]
        x_train_partial = self.x_train[1000:]
        y_train_partial = self.y_train[1000:]

        self.history = self.model.fit(
            x_train_partial, y_train_partial, epochs=epochs, batch_size=batch_size,
            validation_data=(x_val, y_val)
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

    def plot_accuracy(self):
        """Plot training and validation accuracy"""
        history_dict = self.history.history
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1, len(acc_values) + 1)

        plt.plot(epochs, acc_values, 'b-.', label='Training Accuracy')
        plt.plot(epochs, val_acc_values, 'b-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        """Evaluate the model"""
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)
