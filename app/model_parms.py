# Reference to official docs:
# Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Gradient Boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# TensorFlow: https://www.tensorflow.org/guide/keras/sequential_model

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def lr_param():
    solver = st.selectbox(
        "Solver:", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
    )

    if solver in ["newton-cg", "lbfgs", "sag"]:
        penalties = ["l2", "none"]
    elif solver == "saga":
        penalties = ["l1", "l2", "none", "elasticnet"]
    elif solver == "liblinear":
        penalties = ["l1", "l2"]

    penalty = st.selectbox("Penalty:", options=penalties)
    C = st.number_input("Regularization Strength (C):", 0.1, 2.0, 1.0, 0.1)
    C = np.round(C, 3)
    max_iter = st.number_input("Max Iterations:", value=500, min_value=1)
    class_weight = st.selectbox("Class Weight:", options=["balanced", None])
    tol = st.number_input("Tolerance:", value=0.0001, format="%.6f")

    params = {
        "solver": solver,
        "penalty": penalty,
        "C": C,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "tol": tol,
    }

    lr_model = LogisticRegression(**params)
    return lr_model


def rf_param():
    """Random Forest"""
    n_estimators = st.number_input(
        "Number of Estimators:", min_value=1, max_value=500, value=50, step=1
    )
    max_depth = st.number_input(
        "Max Depth of Trees:", min_value=1, max_value=50, value=10, step=1
    )
    min_samples_split = st.number_input(
        "Min Samples Split:", min_value=1, value=10, step=1
    )
    min_samples_leaf = st.number_input(
        "Min Samples Leaf:", min_value=1, value=5, step=1
    )
    class_weight = st.selectbox("Class Weight:", options=["balanced", None])
    max_features = st.selectbox("Max Features:", options=[None, "auto", "sqrt", "log2"])
    criterion = st.selectbox("Criterion:", options=["gini", "entropy"])
    random_state = st.number_input("Random State:", value=42, step=1)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
        "max_features": max_features,
        "criterion": criterion,
        "random_state": random_state,
    }

    rf_model = RandomForestClassifier(**params)
    return rf_model


def gb_param():
    """Gradient Boosting"""
    n_estimators = st.number_input(
        "Number of Estimators:", min_value=1, max_value=500, value=50, step=1
    )
    max_depth = st.number_input(
        "Max Depth of Estimators:", min_value=1, max_value=10, value=3, step=1
    )
    learning_rate = st.number_input(
        "Learning Rate:", min_value=0.001, max_value=0.5, value=0.1, step=0.01
    )
    random_state = st.number_input("Random State:", value=42, step=1)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
    }

    gb_model = GradientBoostingClassifier(**params)
    return gb_model


def svc_param():
    """SVC classifier"""
    kernel = st.selectbox("Kernel:", options=["linear", "poly", "rbf", "sigmoid"])
    C = st.number_input(
        "Regularization Strength (C):",
        min_value=0.01,
        max_value=2.0,
        value=0.5,
        step=0.01,
    )
    class_weight = st.selectbox("Class Weight:", options=["balanced", None])
    random_state = st.number_input("Random State:", value=42, step=1)
    params = {
        "kernel": kernel,
        "C": C,
        "class_weight": class_weight,
        "random_state": random_state,
        "probability": True,
    }
    svm_model = SVC(**params)
    return svm_model


def tf_param():
    """TensorFlow model parameters"""
    epochs = st.number_input("Number of Epochs:", min_value=1, value=60, step=1)

    use_batch_size = st.checkbox("Specify Batch Size")
    if use_batch_size:
        batch_size = st.number_input("Batch Size:", min_value=1, value=32, step=1)
    else:
        batch_size = None

    optimizer = st.checkbox("Use Learning Rate")
    if optimizer:
        learning_rate = st.number_input(
            "Learning Rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001
        )
    else:
        learning_rate = None

    dropout_rate = st.number_input(
        "Dropout Rate:", min_value=0.0, max_value=0.5, value=0.3, step=0.01
    )

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
    }

    return params
