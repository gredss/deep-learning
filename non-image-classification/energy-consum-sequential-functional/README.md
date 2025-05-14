# Energy Consumption Prediction Model

This repository contains the code for building and tuning a neural network model to predict energy consumption based on various input features. The goal of the model is to minimize prediction errors and optimize performance using hyperparameter tuning.

## Table of Contents

* [Overview](#overview)
* [Model Description](#model-description)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Model Performance](#model-performance)
* [Installation](#installation)
* [Usage](#usage)
* [License](#license)

---

## Overview

The project focuses on predicting energy consumption in a given facility using machine learning. The model uses a deep neural network, with hyperparameters tuned for optimal performance. The implementation leverages Keras and Keras Tuner for model building and hyperparameter optimization.

---

## Model Description

The model is built using a functional API in Keras, with the following key components:

* **Input Layer**: The input consists of a feature vector representing various parameters affecting energy consumption.
* **Hidden Layers**: The model includes multiple hidden layers with configurable neurons and activation functions (ReLU, Tanh, or SELU), regularization (L2), batch normalization, and dropout.
* **Output Layer**: A single neuron with a linear activation function to predict energy consumption.

The model is compiled using different optimizers and loss functions depending on the hyperparameters chosen during the tuning process.

---

## Hyperparameter Tuning

The hyperparameters of the model are optimized using `Keras Tuner`, specifically the `RandomSearch` method. The following hyperparameters are tuned:

* **Number of Layers**: Between 1 and 3.
* **Units per Layer**: Between 32 and 256.
* **Activation Functions**: ReLU, Tanh, or SELU.
* **L2 Regularization**: Values of 0.00001, 0.0001, or 0.001.
* **Batch Normalization**: Optional for each layer.
* **Dropout Rate**: Between 0.0 and 0.5.
* **Optimizer**: Adam or RMSprop.
* **Learning Rate**: Ranges from 1e-4 to 1e-2.
* **Loss Function**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

The best hyperparameters are selected based on validation loss, and the final model is built with those parameters.

### Example of Hyperparameter Tuning Code:

```python
from keras_tuner import RandomSearch

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='energy_model_tuning'
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
```

---

## Model Performance

### Baseline Models

The performance of the model is evaluated using the following metrics:

* **R² Score**: Measures the goodness of fit of the model.
* **Root Mean Squared Error (RMSE)**: Measures the prediction error.

Below are the results for baseline models:

* **Sequential Model**:

  * R² Score: 0.298
  * RMSE: 7.55
* **Functional Model**:

  * R² Score: 0.281
  * RMSE: 7.64

### Tuned Models

After tuning the hyperparameters, the following results were obtained for the tuned sequential models:

* **Model 1**:

  * Best Train RMSE: 2.7382
  * Best Val RMSE: 2.7469
  * R² Score: -0.0013
* **Model 2**:

  * Best Train RMSE: 2.4840
  * Best Val RMSE: 2.4978
  * R² Score: 0.3099
* **Model 3**:

  * Best Train RMSE: 3.1697
  * Best Val RMSE: 2.5951
  * R² Score: 0.1929

---

## Installation

To install the required libraries for the project, you can use the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes dependencies such as:

* `keras`
* `keras-tuner`
* `numpy`
* `matplotlib`
* `scikit-learn`

---

## Usage

To train the model, run the following code:

1. Load the dataset and preprocess it.
2. Define and build the model using the provided function `build_model()`.
3. Initialize the `RandomSearch` tuner and start the hyperparameter tuning.
4. After tuning, use the best model for predictions and evaluate the performance.

Example:

```python
# Load your data
X_train, y_train = load_data()

# Build the model using the tuner
tuner.search(X_train, y_train)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.evaluate(X_test, y_test)
```

---

## License

This project is for learning purpose.

