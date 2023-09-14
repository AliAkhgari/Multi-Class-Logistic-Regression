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

## Usage

```python
import pandas as pd
from logistic_regression import LogisticRegressionOneVsRest
from sklearn.model_selection import train_test_split


df = pd.read_csv("Data/penguins.csv")
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(columns="index", inplace=True)

logistic_reg = LogisticRegressionOneVsRest(standardization=True)

train_data, test_data = train_test_split(df, random_state=12, train_size_ratio=0.7)

x_train = train_data[
    ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
]
y_train = train_data["species"]

x_test = test_data[
    ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
]
y_test = test_data["species"]

logistic_reg.fit(x=x_train, y=y_train, iteration=1000, lr=0.1)

y_pred = logistic_reg.predict(x=x_test)
