from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def parallel_cross_val(model, X_train, y_train, cv=5):
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)


def train_model(model, X_train, y_train, model_name=None, cv=5):
    """Train the model with cross-validation and return the trained model."""
    cv_scores = parallel_cross_val(model, X_train, y_train, cv=cv)
    print(f"\n{model_name} Cross-Validation Results:" if model_name else "\nCross-Validation Results:")
    print(f"Mean F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    model.fit(X_train, y_train)
    return model, cv_scores


def validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5, is_tf_model=False):
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
