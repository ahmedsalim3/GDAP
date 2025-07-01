###################################################################################################################################
# A class to handle TensorFlow model creation, training, validation, and evaluation.
###################################################################################################################################

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


class TensorFlowModel:
    @staticmethod
    def to_constants(_state):
        """Converts the data in the session state dictionary to TensorFlow constants."""
        return {
            "X_train": tf.constant(_state["X_train"]),
            "y_train": tf.constant(_state["y_train"]),
            "X_test": tf.constant(_state["X_test"]),
            "y_test": tf.constant(_state["y_test"]),
            "X_val": tf.constant(_state["X_val"]),
            "y_val": tf.constant(_state["y_val"]),
        }

    @staticmethod
    def train(X_train, y_train, X_test, y_test, tf_params):
        """Trains a TensorFlow model using the provided training data and parameters."""
        epochs = tf_params["epochs"]
        batch_size = tf_params["batch_size"]
        dropout_rate = tf_params["dropout_rate"]

        model = Sequential(
            [
                Dense(64, activation="relu"),
                Dropout(dropout_rate),
                Dense(128, activation="relu"),
                Dropout(dropout_rate),
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid"),  # Binary classification output
            ]
        )

        if tf_params["learning_rate"] is not None:
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=tf_params["learning_rate"]),
                metrics=["accuracy"],
            )
        else:
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


        if batch_size is not None:
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                verbose=0,
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=0,
            )

        loss, acc = model.evaluate(X_test, y_test)

        return model, history, acc, loss

    @staticmethod
    def valid(_state, threshold=0.5):
        from gdap.models.model_training import validate_model

        return validate_model(
            model=_state["classifier"],
            X_test=_state["X_test"],
            y_test=_state["y_test"],
            X_val=_state["X_val"],
            y_val=_state["y_val"],
            threshold=threshold,
            is_tf_model=True,
        )

    @staticmethod
    def evaluate(_state, threshold, data=None) -> None:
        from tools.model_evaluator import ModelEvaluator

        if data == "Validation Data":
            X = _state["X_val"]
            Y = _state["y_val"]

        elif data == "Test Data":
            X = _state["X_test"]
            Y = _state["y_test"]

        Evaluation = ModelEvaluator(
            model=_state["classifier"],
            X=X,
            Y=Y,
            threshold=threshold,
            model_name=f'{_state["previous_classifier"]} Model\non {_state["disease_name"]} Disease',
            figsize=(12, 14),
            is_tf_model=True,
            history=_state["history"],
            history_figsize=(12, 5),
        )
        Evaluation.plot_history()
        Evaluation.plot_evaluation()
