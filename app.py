# ==========================================
# Iris Flower Classification using Simple RNN
# One-to-One RNN
# ==========================================

# Import Libraries

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# ==========================================
# Configuration
# ==========================================

MODEL = "iris_rnn.keras"

SCALER = "scaler.pkl"

ENCODER = "label_encoder.pkl"

TIME_STEPS = 4

FEATURES = 1

# ==========================================
# Load Dataset
# ==========================================

def load_dataset():

    print("Loading Iris Dataset...")

    iris = load_iris()

    X = iris.data

    y = iris.target

    target_names = iris.target_names

    print("Feature Shape :", X.shape)

    print("Target Shape :", y.shape)

    return X, y, target_names

# ==========================================
# Prepare Data
# ==========================================

def prepare_data():

    X, y, target_names = load_dataset()

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    with open(SCALER, "wb") as f:
        pickle.dump(scaler, f)

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    with open(ENCODER, "wb") as f:
        pickle.dump(encoder, f)

    y = to_categorical(y)

    # Reshape for RNN
    X = X.reshape(
        X.shape[0],
        TIME_STEPS,
        FEATURES
    )

    print("Reshaped X :", X.shape)

    print("Y Shape :", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.2,
        random_state=42

    )

    return (

        X_train,
        X_test,
        y_train,
        y_test,
        target_names

    )
# ==========================================
# Train Model
# ==========================================

def train_model():

    print("\nPreparing Data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        target_names
    ) = prepare_data()

    print("\nBuilding Model...")

    model = Sequential()

    model.add(

        SimpleRNN(

            64,

            input_shape=(TIME_STEPS, FEATURES)

        )

    )

    model.add(

        Dense(

            32,

            activation="relu"

        )

    )

    model.add(

        Dense(

            3,

            activation="softmax"

        )

    )

    model.summary()

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("\nTraining Model...\n")

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=30,

        batch_size=16,

        verbose=1

    )

    print("\nSaving Model...")

    model.save(MODEL)

    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=1

    )

    print("\nTest Accuracy :", accuracy)

    predictions = model.predict(

        X_test,

        verbose=0

    )

    y_pred = np.argmax(

        predictions,

        axis=1

    )

    y_true = np.argmax(

        y_test,

        axis=1

    )

    print("\nClassification Report\n")

    print(

        classification_report(

            y_true,

            y_pred,

            target_names=target_names

        )

    )

    print(

        "\nConfusion Matrix\n"

    )

    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )

    print("\nModel Training Completed Successfully.")
    # ==========================================
# Predict Flower
# ==========================================

def predict_flower(

    sepal_length,
    sepal_width,
    petal_length,
    petal_width

):

    model = load_model(MODEL)

    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)

    with open(ENCODER, "rb") as f:
        encoder = pickle.load(f)

    sample = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    sample = scaler.transform(sample)

    sample = sample.reshape(
        1,
        TIME_STEPS,
        FEATURES
    )

    prediction = model.predict(
        sample,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    flower = encoder.inverse_transform(
        [predicted_class]
    )[0]

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    return flower_names[flower]


# ==========================================
# Train Model if Not Exists
# ==========================================

if not os.path.exists(MODEL):

    train_model()


# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(

    page_title="Iris Flower Classification",

    page_icon="🌸"

)

st.title("🌸 Iris Flower Classification using Simple RNN")

st.write("### One-to-One RNN")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=0.2
)

if st.button("Predict"):

    prediction = predict_flower(

        sepal_length,

        sepal_width,

        petal_length,

        petal_width

    )

    st.success(

        f"Predicted Flower : {prediction}"

    )