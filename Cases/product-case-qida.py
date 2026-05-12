
# Simple neural network parameter initialization

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
        layer_dims[l], layer_dims[l-1]
        ) * np.sqrt(2 / layer_dims[l-1])

        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    return np.maximum(0,Z), Z

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "sigmoid":
        A = 1/(1+np.exp(-Z))

    cache = (linear_cache, Z)
    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden layers → ReLU
    for l in range(1, L):
        A, cache = linear_activation_forward(
            A,
            parameters["W"+str(l)],
            parameters["b"+str(l)],
            activation="relu"
        )
        caches.append(cache)

    # Output layer → Sigmoid
    AL, cache = linear_activation_forward(
        A,
        parameters["W"+str(L)],
        parameters["b"+str(L)],
        activation="sigmoid"
    )

    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y*np.log(AL+1e-8) + (1-Y)*np.log(1-AL+1e-8)) / m
    return np.squeeze(cost)

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    return dA * s * (1-s)

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # Initial gradient
    dAL = -(np.divide(Y, AL+1e-8) - np.divide(1-Y, 1-AL+1e-8))

    # Last layer (SIGMOID)
    current_cache = caches[L-1]
    linear_cache, activation_cache = current_cache

    s = 1/(1+np.exp(-activation_cache))
    dZ = dAL * s * (1-s)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    grads["dW"+str(L)] = np.dot(dZ, A_prev.T) / m
    grads["db"+str(L)] = np.sum(dZ, axis=1, keepdims=True) / m

    dA_prev = np.dot(W.T, dZ)

    # Hidden layers (ReLU-like behavior simplified)
    for l in reversed(range(L-1)):

        linear_cache, Z = caches[l]

        dZ = np.array(dA_prev, copy=True)
        dZ[Z <= 0] = 0   # correcto ahora porque Z es pre-activación ReLU

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        grads["dW"+str(l+1)] = np.dot(dZ, A_prev.T) / m
        grads["db"+str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / m

        dA_prev = np.dot(W.T, dZ)

    return grads

def update_parameters(parameters, grads, learning_rate):
    parameters = parameters.copy()

    L = len(parameters)//2

    for l in range(L):
        parameters["W"+str(l+1)] -= learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] -= learning_rate * grads["db"+str(l+1)]

    return parameters

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(
    "/Users/franciscofrias/Documents/Qida/table-byai.csv" 
)

# -----------------------------
# Feature / Target separation
# -----------------------------

X = df.drop(columns=["converted"])
y = df["converted"]

# -----------------------------
# Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

# 1️⃣ Aprendé media y std SOLO del train
X_train_scaled = scaler.fit_transform(X_train)

# 2️⃣ Usá ESA MISMA media y std para el test
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Baseline Model: Logistic Regression
# -----------------------------

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

# Predictions
y_pred_prob = baseline_model.predict_proba(X_test)[:, 1]
y_pred = baseline_model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------

roc_auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

print("Baseline Model Results")
print("-----------------------")
print("ROC-AUC:", roc_auc)
print("Accuracy:", accuracy)


import numpy as np

# Convert dataframe to numpy arrays
X_train_np = X_train_scaled.T
Y_train_np = y_train.values.reshape(1, -1)

X_test_np = X_test_scaled.T
Y_test_np = y_test.values.reshape(1, -1)

# Network architecture (demo version)
layers_dims = [X_train_np.shape[0], 16, 8, 1]

# Initialize parameters
parameters = initialize_parameters_deep(layers_dims)

# Training hyperparameters
learning_rate = 0.01
num_iterations = 1500

print("\nNeural Network Training")
print("-----------------------")

# Training loop
for i in range(num_iterations):

    # Forward propagation
    AL, caches = L_model_forward(X_train_np, parameters)

    # Compute cost
    cost = compute_cost(AL, Y_train_np)

    # Backpropagation
    grads = L_model_backward(AL, Y_train_np, caches)

    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print cost every 100 iterations (for demo visibility)
    if i % 100 == 0:
        print(f"Iteration {i} - Cost: {cost:.4f}")

print("Training finished!")

for key in parameters:
    print(key, parameters[key])

# Evaluar red en test
AL_test, _ = L_model_forward(X_test_np, parameters)

y_pred_prob_nn = AL_test.flatten()
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

roc_auc_nn = roc_auc_score(y_test, y_pred_prob_nn)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

print("\nNeural Network Test Results")
print("----------------------------")
print("ROC-AUC:", roc_auc_nn)
print("Accuracy:", accuracy_nn)


# Question mapping for display
question_mapping = {
    "q1_urgency": "When do you need to start the service?",
    "q2_situation": "What is the patient's current situation?",
    "q3_hours": "How many hours of care per day do you estimate needing?",
    "q4_complexity": "What type of support does the patient need?",
    "q5_previous_service": "Have you used a professional home care service before?"
}

# Reverse mapping from encoded values to human-readable answers

decode_mapping = {

    "q1_urgency": {
        4: "Within the next 24-72 hours",
        3: "Within the next week",
        2: "Within the next 2-4 weeks",
        1: "I am just exploring options"
    },

    "q2_situation": {
        4: "Is in the hospital and will be discharged soon",
        3: "Lives alone and far away",
        2: "Lives alone and close to us",
        1: "Lives with family"
    },

    "q3_hours": {
        4: "24h",
        3: "8-12h",
        2: "1-6h",
        1: "Not sure"
    },

    "q4_complexity": {
        4: "More complex care (medication, medical conditions)",
        3: "Help with hygiene and mobility",
        2: "Shopping and meal preparation",
        1: "Companionship and supervision"
    },

    "q5_previous_service": {
        4: "Yes, currently",
        3: "Yes, in the past",
        2: "No, this would be the first time",
        1: "We are unsure if it is the right option"
    }
}

# -----------------------------
# Test prediction on new lead
# -----------------------------

def predict_new_lead(parameters, lead_features, scaler):

    # Crear DataFrame con mismos nombres de columnas
    X_new_df = pd.DataFrame(
        [lead_features],
        columns=X_train.columns
    )

    # Escalar correctamente
    X_new_scaled = scaler.transform(X_new_df)

    X_new_np = X_new_scaled.T

    AL, _ = L_model_forward(X_new_np, parameters)

    probability = AL[0][0]

    return probability

print("Scaler mean:", scaler.mean_)
print("Scaler std:", scaler.scale_)

# Example new lead (simulate high quality lead)

print("\nNew Lead Prediction")
print("-----------------------")

new_lead = [2, 2, 3, 4, 3] 

# Randomly generate a new lead within valid encoding ranges
#new_lead = [ random.randint(1, 4),  # q1_urgency
#    random.randint(1, 4),  # q2_situation
#    random.randint(1, 4),  # q3_hours
#    random.randint(1, 4),  # q4_complexity
#    random.randint(1, 4)   # q5_previous_service
#]

feature_names = list(question_mapping.keys())

print("\nLead Profile:")

for feature_name, encoded_value in zip(feature_names, new_lead):

    question_text = question_mapping[feature_name]

    answer_text = decode_mapping[feature_name].get(
        encoded_value,
        "Unknown answer"
    )

    print(question_text)
    print("Answer:", answer_text)
    print("")
    
prob = predict_new_lead(parameters, new_lead, scaler)

print("Lead qualification probability:", prob)

if prob > 0.7:
    print("👉 High qualified lead")
else:
    print("👉 Low qualified lead")