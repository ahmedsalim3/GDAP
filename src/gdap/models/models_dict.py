from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sklearn_models = {
    "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random_Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "Gradient_Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    "SVM": SVC(
        kernel="linear",
        C=0.5,
        class_weight="balanced",
        random_state=42,
        probability=True,
    ),
}
