import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Placement Prediction API",
    description="Placement prediction using Perceptron and Logistic Regression",
    version="1.0.0"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("placement.csv")


# ============================================================
# PREPROCESSING
# ============================================================

df_model = df.drop(
    columns=["Name", "Roll No", "Gender"]
)

X = df_model.drop(columns=["Placed"])
y = df_model["Placed"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]


X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()


# ============================================================
# ENCODING
# ============================================================

binary_mapping = {
    "No": 0,
    "Yes": 1
}

X_train["Internships"] = X_train["Internships"].map(
    binary_mapping
)

X_test["Internships"] = X_test["Internships"].map(
    binary_mapping
)

X_train["Certifications"] = X_train["Certifications"].map(
    binary_mapping
)

X_test["Certifications"] = X_test["Certifications"].map(
    binary_mapping
)


project_mapping = {
    "Beginner": 0,
    "Moderate": 1,
    "Advanced": 2
}

X_train["Project Level"] = X_train["Project Level"].map(
    project_mapping
)

X_test["Project Level"] = X_test["Project Level"].map(
    project_mapping
)


# ============================================================
# ONE-HOT ENCODE BRANCH
# ============================================================

X_train = pd.get_dummies(
    X_train,
    columns=["Branch"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["Branch"],
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# SCALING
# ============================================================

mean = X_train.mean()
std = X_train.std()

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


X_train_np = X_train_scaled.to_numpy(dtype=float)

y_train_np = y_train.to_numpy(dtype=int)


# ============================================================
# PERCEPTRON
# ============================================================

class Perceptron:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=100
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None


    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(n_samples):

                z = np.dot(
                    X[i],
                    self.weights
                ) + self.bias

                prediction = 1 if z > 0 else 0

                update = self.learning_rate * (
                    y[i] - prediction
                )

                self.weights += update * X[i]

                self.bias += update


    def predict(self, X):

        z = np.dot(
            X,
            self.weights
        ) + self.bias

        return (z > 0).astype(int)


# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

def sigmoid(z):

    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None


    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            # Forward pass
            z = np.dot(
                X,
                self.weights
            ) + self.bias

            probabilities = sigmoid(z)

            # Error
            error = probabilities - y

            # Gradients
            dw = (
                1 / n_samples
            ) * np.dot(
                X.T,
                error
            )

            db = (
                1 / n_samples
            ) * np.sum(error)

            # Gradient descent
            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )


    def predict_probability(self, X):

        z = np.dot(
            X,
            self.weights
        ) + self.bias

        return sigmoid(z)


    def predict(self, X):

        probabilities = self.predict_probability(X)

        return (
            probabilities >= 0.5
        ).astype(int)


# ============================================================
# TRAIN MODELS
# ============================================================

perceptron = Perceptron(
    learning_rate=0.01,
    epochs=100
)

perceptron.fit(
    X_train_np,
    y_train_np
)


logistic_scratch = LogisticRegressionScratch(
    learning_rate=0.01,
    epochs=1000
)

logistic_scratch.fit(
    X_train_np,
    y_train_np
)


logistic_sklearn = LogisticRegression(
    max_iter=1000
)

logistic_sklearn.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# INPUT SCHEMA
# ============================================================

from pydantic import BaseModel, Field
from typing import Literal


class StudentInput(BaseModel):

    branch: Literal[
        "AI_DS", "CSAI", "CSAIML", "CSDS",
        "CSE", "ECE", "EEE", "IT", "ME"
    ]

    cgpa: float = Field(..., ge=0, le=10)

    percentage_10: float = Field(
        ..., ge=0, le=100
    )

    percentage_12: float = Field(
        ..., ge=0, le=100
    )

    project_level: Literal[
        "Beginner",
        "Moderate",
        "Advanced"
    ]

    internships: Literal[
        "Yes",
        "No"
    ]

    certifications: Literal[
        "Yes",
        "No"
    ]

    backlogs: int = Field(
        ..., ge=0
    )


# ============================================================
# PREPARE USER INPUT
# ============================================================

def prepare_input(student):

    data = pd.DataFrame([{

        "Branch": student.branch,

        "CGPA": student.cgpa,

        "10th_percentage":
            student.percentage_10,

        "12th_percentage":
            student.percentage_12,

        "Project Level":
            student.project_level,

        "Internships":
            student.internships,

        "Certifications":
            student.certifications,

        "Backlogs":
            student.backlogs

    }])


    # Binary encoding

    data["Internships"] = data[
        "Internships"
    ].map(binary_mapping)

    data["Certifications"] = data[
        "Certifications"
    ].map(binary_mapping)


    # Project level

    data["Project Level"] = data[
        "Project Level"
    ].map(project_mapping)


    # Branch one-hot encoding

    data = pd.get_dummies(
        data,
        columns=["Branch"],
        dtype=int
    )


    # Same feature order

    data = data.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    # Same scaling

    data_scaled = (
        data - mean
    ) / std


    return data_scaled.to_numpy(
        dtype=float
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Student Placement Prediction API",

        "models": [
            "Perceptron",
            "Logistic Regression (Scratch)",
            "Logistic Regression (Sklearn)"
        ],

        "docs":
            "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(student: StudentInput):

    X_input = prepare_input(student)


    perceptron_prediction = (
        perceptron.predict(X_input)[0]
    )


    scratch_prediction = (
        logistic_scratch.predict(X_input)[0]
    )


    sklearn_prediction = (
        logistic_sklearn.predict(X_input)[0]
    )


    return {

        "Perceptron":
            "Placed"
            if perceptron_prediction == 1
            else "Not Placed",

        "Logistic Regression (Scratch)":
            "Placed"
            if scratch_prediction == 1
            else "Not Placed",

        "Logistic Regression (Sklearn)":
            "Placed"
            if sklearn_prediction == 1
            else "Not Placed"
    }