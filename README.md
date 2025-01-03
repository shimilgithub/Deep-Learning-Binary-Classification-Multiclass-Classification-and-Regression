# Deep Learning with Python - Binary, Multiclass Classification & Regression

This repository contains implementations of binary classification, multiclass classification, and regression tasks. The tasks are performed on the following datasets:

- **Binary Classification**: IMDB Dataset (Sentiment Analysis)
- **Multiclass Classification**: Reuters Dataset (News Category Prediction)
- **Regression**: Boston Housing Dataset (Predicting Home Prices)

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib (for plotting graphs)
- Jupyter Notebook (for code implementation and visualization)

## Implementation
### 1. Binary Classification - IMDB
A class `Imdb` is implemented in `imdb.py` that handles the following tasks:
- **prepare_data()**: Prepares the IMDB dataset for training and testing.
- **build_model()**: Builds the binary classification model using Keras.
- **train()**: Trains the model using the prepared dataset.
- **plot_loss()**: Plots the training and validation loss over epochs.
- **plot_accuracy()**: Plots the training and validation accuracy over epochs.
- **evaluate()**: Evaluates the model on the test dataset to show loss and accuracy.

### 2. Multiclass Classification - Reuters
A class `Reuters` is implemented in `reuters.py` that handles the following tasks:
- **prepare_data()**: Prepares the Reuters dataset for training and testing.
- **build_model()**: Builds the multiclass classification model using Keras.
- **train()**: Trains the model using the prepared dataset.
- **plot_loss()**: Plots the training and validation loss over epochs.
- **plot_accuracy()**: Plots the training and validation accuracy over epochs.
- **evaluate()**: Evaluates the model on the test dataset to show loss and accuracy.

### 3. Regression - Boston Housing
A class `BostonHousing` is implemented in `boston_housing.py` that handles the following tasks:
- **prepare_data()**: Prepares the Boston Housing dataset for training and testing.
- **build_model()**: Builds the regression model using Keras.
- **train()**: Trains the model using the prepared dataset.
- **plot_loss()**: Plots the training and validation loss over epochs.
- **plot_accuracy()**: (This function is included, but for regression, accuracy is not relevant. It will plot loss instead.)
- **evaluate()**: Evaluates the model on the test dataset to show loss.

### Jupyter Notebook
The notebook `notebook.ipynb` showcases the implementations of the classes `Imdb`, `Reuters`, and `BostonHousing`, with the relevant functions being demonstrated.

## How to Run
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/shimilgithub/Deep-Learning-Binary-Classification-Multiclass-Classification-and-Regression.git
    ```

2. Open the Jupyter notebook to see the implementation in action:
    ```bash
    jupyter notebook notebook.ipynb
    ```

3. You can also run the individual Python scripts for each task to make use of the repective class:
    ```bash
    python imdb.py
    python reuters.py
    python boston_housing.py
    ```
