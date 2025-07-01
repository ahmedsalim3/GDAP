import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def train_tf_model(X_train, y_train, X_test, y_test, X_val, y_val, epochs=60):
    """Train a TensorFlow model (Sequential) and return the model and performance metrics."""
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
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    return model, history, acc, loss
