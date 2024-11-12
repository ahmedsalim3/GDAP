from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


models = {
    "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random_Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "Gradient_Boosting": GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    ),
    "SVM": SVC(
        kernel="linear",
        C=0.5,
        class_weight="balanced",
        random_state=42,
        probability=True,
    ),
}


def train_tf_model(X_train, y_train, X_test, y_test, X_val, y_val, epochs=60):
    X_train = tf.constant(X_train)
    y_train = tf.constant(y_train)
    X_test = tf.constant(X_test)
    y_test = tf.constant(y_test)
    X_val = tf.constant(X_val)
    y_val = tf.constant(y_val)

    model = Sequential(
        [
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Training sequential API model...")
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    return model, history, acc, loss


def parallel_cross_val(model, X_train, y_train, cv=5):
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)


def train_model(model, X_train, y_train, model_name=None, cv=5):
    """Train the model with cross-validation and return the trained model."""
    cv_scores = parallel_cross_val(model, X_train, y_train, cv=cv)
    print(
        f"\n{model_name} Cross-Validation Results:"
        if model_name
        else "\nCross-Validation Results:"
    )
    print(f"Mean F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    model.fit(X_train, y_train)
    return model, cv_scores


def validate_model(
    model, X_test, y_test, X_val, y_val, threshold=0.5, is_tf_model=False
):
    """Validate models on test and validation sets and return results."""

    if is_tf_model:
        y_test_pred_proba = model.predict(X_test).flatten()
        y_val_pred_proba = model.predict(X_val).flatten()
    else:
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    y_test_pred = (y_test_pred_proba > threshold).astype(int)
    y_val_pred = (y_val_pred_proba > threshold).astype(int)

    test_results = {
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Test Precision": precision_score(y_test, y_test_pred),
        "Test Recall": recall_score(y_test, y_test_pred),
        "Test F1-score": f1_score(y_test, y_test_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_test_pred_proba),
        "Test PR-AUC": average_precision_score(y_test, y_test_pred_proba),
    }

    val_results = {
        "Val Accuracy": accuracy_score(y_val, y_val_pred),
        "Val Precision": precision_score(y_val, y_val_pred),
        "Val Recall": recall_score(y_val, y_val_pred),
        "Val F1-score": f1_score(y_val, y_val_pred),
        "Val ROC-AUC": roc_auc_score(y_val, y_val_pred_proba),
        "Val PR-AUC": average_precision_score(y_val, y_val_pred_proba),
    }

    return test_results, val_results
