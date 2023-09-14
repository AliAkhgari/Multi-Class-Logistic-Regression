# Multi-Class-Logistic-Regression


This project implements a Multiclass Logistic Regression classifier using the One-vs-Rest (OvR) strategy. It allows you to tackle multiclass classification problems by breaking them down into multiple binary classification tasks.

## Introduction

Multiclass classification involves categorizing data into more than two classes or categories. The OvR method is a common approach for solving multiclass problems using binary classifiers. This project provides a versatile implementation of Multiclass Logistic Regression with the OvR strategy.

## Key Characteristics

- **One-vs-Rest Strategy**: Converts a multiclass problem into multiple binary classification subproblems.
- **Data Preprocessing**: Includes preprocessing steps such as feature standardization and intercept term addition to prepare the input data.
- **Sigmoid Activation**: Utilizes the sigmoid (logistic) activation function to estimate class probabilities.
- **Batch Gradient Descent**: Optimizes model parameters using batch gradient descent, minimizing the logistic loss function.
- **Regularization**: Offers L2 regularization (Ridge regularization) as an option to prevent overfitting.
