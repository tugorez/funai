# Simple Linear Models in NumPy

---

This project features implementations of fundamental linear models built from scratch using **NumPy** exclusively.

The core of this project is a **modular design** centered around a generic `Regression` class. This single class can be adapted to perform various tasks—such as linear regression, binary classification, and multi-class classification—by simply providing it with the appropriate activation and loss functions.

---

## Core Concept: A Modular Approach

The `regression.py` file contains a generic `Regression` base class. This class isn't tied to any specific model; instead, it's a flexible engine that can be configured to create different models:

* **Linear Regression**:
    * **Activation**: Identity (i.e., no activation)
    * **Loss**: Mean Squared Error

* **Logistic (Binary) Regression**:
    * **Activation**: Sigmoid
    * **Loss**: Binary Cross-Entropy

* **Softmax (Multi-Class) Regression**:
    * **Activation**: Softmax
    * **Loss**: Categorical Cross-Entropy

This approach demonstrates how various machine learning models share a common underlying structure for training and prediction.

---

## File Descriptions

* `regression.py`: Contains the generic `Regression` base class. This file is not meant to be run directly but is imported by the others.
* `linear_regression.py`: An example of simple linear regression that learns to fit a line to data points.
* `logistic_regression.py`: An example of binary classification that learns to classify grayscale values as either "dark" or "light".
* `softmax_regression.py`: An example of multi-class classification that learns to classify RGB color values into one of several named colors (e.g., "red", "blue", "yellow").