import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

import os
import sys

# =========================
# PATHS DEL EJERCICIO
# =========================

# Directorio donde está p2.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio Files/release del ejercicio
RELEASE_DIR = os.path.join(BASE_DIR, "Files", "release", "W2A2")

# Agregar release al PYTHONPATH
if RELEASE_DIR not in sys.path:
    sys.path.append(RELEASE_DIR)

# =========================
# IMPORTS DEL EJERCICIO
# =========================

from lr_utils import load_dataset
from public_tests import *

# ================================
# Carga del dataset
# ================================
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# ================================
# Dimensiones del dataset
# ================================
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# ================================
# Flatten + normalización
# ================================
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

print("train_set_x shape:", train_set_x.shape)
print("train_set_y shape:", train_set_y.shape)
print("test_set_x shape:", test_set_x.shape)
print("test_set_y shape:", test_set_y.shape)

# ================================
# Función sigmoid
# ================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ================================
# Inicialización
# ================================
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

initialize_with_zeros_test_1(initialize_with_zeros)
initialize_with_zeros_test_2(initialize_with_zeros)

# ================================
# Propagate
# ================================
def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}
    return grads, cost

propagate_test(propagate)

# ================================
# Optimize
# ================================
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

optimize_test(optimize)

# ================================
# Predict
# ================================
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

predict_test(predict)

# ================================
# Modelo
# ================================
def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations=2000,
    learning_rate=0.5,
    print_cost=False
):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(
        w, b, X_train, Y_train,
        num_iterations, learning_rate, print_cost
    )

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        ))
        print("test accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
        ))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d

model_test(model)

# ================================
# Entrenamiento final
# ================================
logistic_regression_model = model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True
)

# ================================
# Visualización de ejemplo
# ================================
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.axis("off")
print(
    "y = " + str(test_set_y[0, index]) +
    ", predicted = \"" +
    classes[int(logistic_regression_model['Y_prediction_test'][0, index])].decode("utf-8") +
    "\""
)

# ================================
# Learning curve
# ================================
costs = np.squeeze(logistic_regression_model["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate = " + str(logistic_regression_model["learning_rate"]))
plt.show()
