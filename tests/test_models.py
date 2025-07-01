import unittest
from unittest.mock import patch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from gdap.models.model_training import train_model, validate_model
from gdap.models.models_dict import sklearn_models


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(30, 10)
        self.y_test = np.random.randint(0, 2, 30)
        self.X_val = np.random.rand(20, 10)
        self.y_val = np.random.randint(0, 2, 20)

    def test_train_model_logistic_regression(self):
        model, cv_scores = train_model(
            LogisticRegression(), self.X_train, self.y_train, model_name="Logistic_Regression"
        )
        self.assertIsInstance(model, LogisticRegression)
        self.assertIsInstance(cv_scores, np.ndarray)

    def test_validate_model(self):
        model, _ = train_model(
            LogisticRegression(), self.X_train, self.y_train, model_name="Logistic_Regression"
        )
        test_results, val_results = validate_model(
            model, self.X_test, self.y_test, self.X_val, self.y_val, threshold=0.5
        )
        self.assertIsInstance(test_results, dict)
        self.assertIn('Test Accuracy', test_results)
        self.assertIsInstance(val_results, dict)


class TestModelsDict(unittest.TestCase):
    def test_sklearn_models_structure(self):
        expected_models = ['Random_Forest', 'Gradient_Boosting', 'SVM', 'Logistic_Regression']
        for model_name in expected_models:
            self.assertIn(model_name, sklearn_models)

    def test_model_instances(self):
        self.assertIsInstance(sklearn_models['Random_Forest'], RandomForestClassifier)
        self.assertIsInstance(sklearn_models['Gradient_Boosting'], GradientBoostingClassifier)
        self.assertIsInstance(sklearn_models['SVM'], SVC)
        self.assertIsInstance(sklearn_models['Logistic_Regression'], LogisticRegression)


if __name__ == "__main__":
    unittest.main()
