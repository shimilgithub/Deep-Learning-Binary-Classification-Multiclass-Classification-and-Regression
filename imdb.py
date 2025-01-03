"""This script implements binary classification using the IMDB dataset with TensorFlow"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

class Imdb:
    """Binary classification class on the IMDB dataset using a dense neural network."""

    def __init__(self, num_words=10000):
        """Initialize the Imdb class"""
        self.num_words = num_words
        self.model = None

    def prepare_data(self):
        """Load and preprocess IMDB dataset."""
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.num_words)
        self.word_index = imdb.get_word_index()
        
        self.x_train = self.vectorize_sequences(x_train)
        self.x_test = self.vectorize_sequences(x_test)
        self.y_train = np.asarray(y_train).astype('float32')
        self.y_test = np.asarray(y_test).astype('float32')

        # Split validation data
        self.x_val = self.x_train[:10000]
        self.y_val = self.y_train[:10000]
        self.x_train_partial = self.x_train[10000:]
        self.y_train_partial = self.y_train[10000:]

    def vectorize_sequences(self, sequences):
        """Convert sequences to binary matrix form"""
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        """Build a binary classification model for the IMDB dataset."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, epochs=20, batch_size=512):
        """Train the model with the training data"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            tf.keras.callbacks.TensorBoard()
        ]
        self.history = self.model.fit(
            self.x_train_partial, self.y_train_partial, epochs=epochs, batch_size=batch_size,
            validation_data=(self.x_val, self.y_val), callbacks=callbacks
        )

    def plot_loss(self):
        """Plot training and validation loss"""
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
        """Evaluate the model """
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)
