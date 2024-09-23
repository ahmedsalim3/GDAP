from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from src.model_evaluation import TfEvaluation, ModelEvaluation


models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=500),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    ),
    "SVM": SVC(
        kernel='linear',
        C=0.5,
        class_weight='balanced',
        random_state=42,
        probability=True
    )
}


def train_tf_model(X_train, y_train, X_test, y_test, X_val, y_val):
    X_train = tf.constant(X_train)
    y_train = tf.constant(y_train)
    X_test = tf.constant(X_test)
    y_test = tf.constant(y_test)
    X_val = tf.constant(X_val)
    y_val = tf.constant(y_val)

    model = Sequential([
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print('Training sequential API model...')
    history = model.fit(
        X_train,
        y_train,
        epochs=60,
        validation_data=(X_test, y_test),
        verbose=0
    )

    model_name = 'Sequential'
    print(f"\n{model_name} Evaluation Results")
    res = model.evaluate(X_test, y_test)
    print(f"Test Loss: {res[0]:.4f}, Test Accuracy: {res[1]:.4f}")

    y_test_pred = (model.predict(X_test) >= 0.5).astype(int)
    test_results = {
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test Precision': precision_score(y_test, y_test_pred, zero_division=1),
        'Test Recall': recall_score(y_test, y_test_pred, zero_division=1),
        'Test F1-score': f1_score(y_test, y_test_pred, zero_division=1)
    }
    
    print(f"\n{model_name} (Test Set):")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    y_val_pred = (model.predict(X_val) >= 0.5).astype(int)
    val_results = {
        'Val Accuracy': accuracy_score(y_val, y_val_pred),
        'Val Precision': precision_score(y_val, y_val_pred, zero_division=1),
        'Val Recall': recall_score(y_val, y_val_pred, zero_division=1),
        'Val F1-score': f1_score(y_val, y_val_pred, zero_division=1)
    }
    
    print(f"\n{model_name} (Validation Set):")
    for metric, value in val_results.items():
        print(f"{metric}: {value:.4f}")
    
    Evaluation = TfEvaluation(model, X_val, y_val, f'{model_name} (Validation Set)', history)
    Evaluation.plot_history()        
    Evaluation.plot_evaluation()
    
    return model, test_results, val_results
        

def parallel_cross_val(model, X_train, y_train, cv=5):
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

def train_model(model, X_train, y_train, cv=5, model_name=None):
    """Train the model with cross-validation and return the trained model."""
    cv_scores = parallel_cross_val(model, X_train, y_train, cv=cv)
    print(f"\n{model_name} Cross-Validation Results:" if model_name else "\nCross-Validation Results:")
    print(f"Mean F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    model.fit(X_train, y_train)
    return model

def validate_model(model, X_test, y_test, X_val, y_val, model_name=None): #, threshold=0.5
    """Validate the model on test and validation sets and return results."""
    y_test_pred = model.predict(X_test) # > threshold
    test_results = {
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test Precision': precision_score(y_test, y_test_pred, zero_division=1),
        'Test Recall': recall_score(y_test, y_test_pred, zero_division=1),
        'Test F1-score': f1_score(y_test, y_test_pred, zero_division=1)
    }
    
    print(f"\n{model_name} (Test Set):" if model_name else "\n(Test Set):")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    if hasattr(model, "predict_proba"):
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_pred_proba >= 0.5).astype(int)  # Threshold is set to 0.5
        val_results = {
            'Val Accuracy': accuracy_score(y_val, y_val_pred),
            'Val Precision': precision_score(y_val, y_val_pred, zero_division=1),
            'Val Recall': recall_score(y_val, y_val_pred, zero_division=1),
            'Val F1-score': f1_score(y_val, y_val_pred, zero_division=1),
            'Val AUC': roc_auc_score(y_val, y_val_pred_proba)
        }
        
        print(f"\n{model_name} (Validation Set):" if model_name else "\n(Validation Set):")
        for metric, value in val_results.items():
            print(f"{metric}: {value:.4f}")
    
    Evaluation = ModelEvaluation(model, X_val, y_val, f'{model_name} (Validation Set)' if model_name else '(Validation Set)')    
    Evaluation.plot_evaluation()
        
    return test_results, val_results



def cross_validate_and_evaluate(model, X_train, y_train, X_test, y_test, X_val, y_val, model_name=None, cv=5):
    
    model = train_model(model, X_train, y_train, cv, model_name)
    test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, model_name)
        
    return model, test_results, val_results