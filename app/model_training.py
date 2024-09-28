import streamlit as st
from src.ml_models import train_model, validate_model


def train_classifier(model, model_name=None):
    if "X_train" in st.session_state and "y_train" in st.session_state:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        classifier, cv_scores = train_model(model, X_train, y_train, model_name)
    else:
        return None, None, True
    return classifier, cv_scores, False


def validate_classifier(classifier, threshold, is_tf_model=False):

    if (
        "X_test" in st.session_state
        and "y_test" in st.session_state
        and "X_val" in st.session_state
        and "y_val" in st.session_state
        and "classifier" in st.session_state
    ):
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        X_val = st.session_state["X_val"]
        y_val = st.session_state["y_val"]

        test_results, val_results = validate_model(
            classifier, X_test, y_test, X_val, y_val, threshold=threshold, is_tf_model=is_tf_model
        )
    else:
        return None, None, True
    return test_results, val_results, False


def train_tf(X_train, y_train, X_test, y_test, tf_params):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    epochs = tf_params['epochs']
    batch_size = tf_params['batch_size']
    dropout_rate = tf_params['dropout_rate']

    model = Sequential(
        [
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    if tf_params['learning_rate'] is not None:
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=tf_params['learning_rate']), metrics=["accuracy"])
    else:
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Training sequential API model...")
    
    if batch_size is not None:
        history = model.fit(
            X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, verbose=0
        )
    else:
        history = model.fit(
            X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0
        )
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    return model, history, acc, loss

