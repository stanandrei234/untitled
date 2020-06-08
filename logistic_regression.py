import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def L(X, W, B):
    return 1 / (1 + np.exp(
        (-1) * (
                X * W.T + B
        )
    ))


def cost_function(Y, YY):
    # number of elements
    n = len(Y)

    # calculate the cost function between actual and predicted
    return 1 / n * sum(Y * np.log(YY) + (1 - Y) * np.log(1 - YY))


def gradient_descent(X, W, B, Y):
    n = len(X)
    f = L(X, W, B)  # functis prezisa
    cost = cost_function(Y, f)  # eroarea dintre valoarea prezisa si cea actuala

    # gradient calculation

    dw = (1 / n) * (np.dot(X.T, (f - Y.T).T))
    db = (1 / n) * (np.sum(f - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def predict(X, Y, W, B, iteratii, learning_rate):
    costs = []

    for i in range(iteratii):
        grads, cost = gradient_descent(X, W, B, Y)
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #

        if (i % 100 == 0):
            costs.append(cost)
            # print("Cost after %i iteration is %f" %(i, cost))

        # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs


df = pd.read_csv('iris-data.csv')
print(df.head())
