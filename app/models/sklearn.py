###################################################################################################################################
# A class to handle different sklearn classifiers, model training, validation, and evaluation.
###################################################################################################################################

from gene_disease.models.model_training import train_model, validate_model


class SkLearn:

    @staticmethod
    def classifier(_state):
        classifier_options = _state["classifier_options"]
        parms = _state["parms"]

        # Only import the necessary classifier when needed
        if classifier_options == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**parms)
        elif classifier_options == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**parms)
        elif classifier_options == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(**parms)
        elif classifier_options == "SVC":
            from sklearn.svm import SVC

            return SVC(**parms)

    @staticmethod
    def train(model, _state):
        return train_model(
            model,
            _state["X_train"],
            _state["y_train"],
            model_name=_state["classifier_options"],
            cv=5,
        )

    @staticmethod
    def valid(_state, threshold=0.5):
        return validate_model(
            model=_state["classifier"],
            X_test=_state["X_test"],
            y_test=_state["y_test"],
            X_val=_state["X_val"],
            y_val=_state["y_val"],
            threshold=threshold,
            is_tf_model=False,
        )

    @staticmethod
    def evaluate(_state, threshold, data=None):
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
        )
        Evaluation.plot_history()
        Evaluation.plot_evaluation()
