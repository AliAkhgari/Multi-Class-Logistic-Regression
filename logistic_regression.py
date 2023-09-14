import copy

import numpy as np
import pandas as pd


class LogisticRegressionOneVsRest:
    def __init__(self, standardization: bool) -> None:
        self.theta = {}
        self.standardization = standardization

    @staticmethod
    def y_preprocess(
        y: pd.Series,
        label_name: str,
    ):

        y = pd.DataFrame(y)

        for i in range(len(y[label_name].unique())):
            y[str(i)] = np.nan

        for i in range(len(y[label_name].unique())):
            for index, row in y.iterrows():
                if row[label_name] == y[label_name].unique()[i]:
                    y.loc[index, str(i)] = 1
                else:
                    y.loc[index, str(i)] = 0

        return y

    def x_preprocess(self, x: pd.DataFrame):
        # FIXME: adding ones for intercept with pandas has different results with adding ones with numpy!!!!
        x = copy.deepcopy(x)
        cols = x.columns
        if self.standardization:
            for col in cols:
                x[col] = (x[col] - np.mean(x[col])) / np.std(x[col])

        x["intercept"] = 1.0
        new_cols = ["intercept"]
        new_cols.extend(cols)
        x = x[new_cols]

        return x

    @staticmethod
    def h_theta(x: np.ndarray, theta: np.ndarray):
        z = np.dot(x, theta)
        return 1.0 / (1.0 + np.exp(-z))

    def batch_gradient_descent(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iteration: int = 1000,
        lr: float = 0.01,
        alpha: float = 1.0,
        regularization: bool = False,
    ):
        n = x.shape[0]
        d = x.shape[1]
        y = y.reshape((n, 1))

        theta = np.zeros((d, 1))
        for _ in range(iteration):
            if regularization:
                gradient = (
                    np.dot(x.T, (self.h_theta(x=x, theta=theta) - y)) + alpha * theta
                ) / n
            else:
                gradient = np.dot(x.T, (self.h_theta(x=x, theta=theta) - y)) / n
            theta = theta - lr * gradient.reshape((d, 1))

        return theta

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        iteration: int = 1000,
        lr: float = 0.001,
        alpha: float = 1.0,
        regularization: bool = False,
    ):
        label_name = y.name
        y = self.y_preprocess(y, label_name=label_name)
        x = self.x_preprocess(x=x)
        x = x.to_numpy()

        for i in range(len(y[label_name].unique())):
            y_arr = y[str(i)].to_numpy()
            self.theta[y[label_name].unique()[i]] = self.batch_gradient_descent(
                x=x,
                y=y_arr,
                iteration=iteration,
                lr=lr,
                alpha=alpha,
                regularization=regularization,
            )

    def predict(self, x):
        x = self.x_preprocess(x=x)
        x = x.to_numpy()

        y_pred = []
        for i in range(x.shape[0]):
            pred = None
            max_prob = -1
            for label in self.theta.keys():
                prob = self.h_theta(x[i], self.theta[label])
                if prob > max_prob:
                    max_prob = prob
                    pred = label
            y_pred.append(pred)

        return y_pred
