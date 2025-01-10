###################################################################################################################################
# A class for evaluating and visualizing machine learning model performance using various metrics.
#
#   It provides methods to calculate and plot confusion matrix, ROC curve, precision-recall curve, 
#   classification report, and training history. Supports both scikit-learn and TensorFlow models
###################################################################################################################################

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


class ModelEvaluator:
    """
    A class for evaluating and visualizing machine learning models using various metrics.

    Attributes:
        model: The trained model to evaluate.
        X: Test features (e.g., testing or validation sets).
        Y: True labels for the test set (the labels corresponding to the features).
        threshold: Probability threshold for classification (default is 0.5).
        model_name: Optional name of the model being evaluated for the figure title.
        figsize: Size of the figure for plotting evaluation (default is (14, 12)).
        is_tf_model: Boolean indicating if the model is a TensorFlow model (default is False).
            - If `True`, the model does not predict probabilities; features will be
              flattened, and predictions are made based on whether the predicted
              value meets or exceeds the threshold.
            - If `False`, for models like Scikit-Learn that do predict probabilities,
              the probability of the positive class (`[:, 1]`) is calculated before
              checking if it meets or exceeds the threshold.

        history: History logs for TensorFlow model.
        history_figsize: Size of the figure for plotting training history (default is (14, 5)).
    """

    def __init__(
        self,
        model,
        X,
        Y,
        threshold=0.5,
        model_name=None,
        figsize=(14, 12),
        is_tf_model=False,
        history=None,
        history_figsize=(14, 5),
    ):
        self.model = model
        self.X = X
        self.Y = Y
        self.model_name = model_name
        self.history = history
        self.threshold = threshold
        self.figsize = figsize
        self.history_figsize = history_figsize
        self.primaryColor = "#4fd8a0"  # Light green (primary color)
        self.secondaryColor = "#1f3a60" 

        if is_tf_model:
            self.y_pred_proba = self.model.predict(self.X).flatten()
        else:
            self.y_pred_proba = self.model.predict_proba(self.X)[:, 1]

        self.y_pred = (self.y_pred_proba >= self.threshold).astype(int)

    def _conf_matrix(self):
        cm = confusion_matrix(self.Y, self.y_pred)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(len(cm)):
            if total_actual[i] > 0:
                cm_percent[i, i] = true_positives[i] / total_actual[i]

        return cm, cm_percent

    def _roc_curve_auc(self):
        fpr_1, tpr_1, _ = roc_curve(self.Y, self.y_pred_proba)
        roc_auc_1 = auc(fpr_1, tpr_1)

        fpr_0, tpr_0, _ = roc_curve(1 - self.Y, 1 - self.y_pred_proba)
        roc_auc_0 = auc(fpr_0, tpr_0)

        return {
            "class_1": (fpr_1, tpr_1, roc_auc_1),
            "class_0": (fpr_0, tpr_0, roc_auc_0),
        }

    def _pr_curve_auc(self):
        precision_1, recall_1, _ = precision_recall_curve(
            self.Y, self.y_pred_proba
        )
        pr_auc_1 = average_precision_score(self.Y, self.y_pred_proba)

        precision_0, recall_0, _ = precision_recall_curve(
            1 - self.Y, 1 - self.y_pred_proba
        )
        pr_auc_0 = average_precision_score(1 - self.Y, 1 - self.y_pred_proba)

        return {
            "class_1": (precision_1, recall_1, pr_auc_1),
            "class_0": (precision_0, recall_0, pr_auc_0),
        }

    def _class_report_df(self):
        class_report = classification_report(self.Y, self.y_pred, output_dict=True)
        report_data = []
        for label, metrics in class_report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                row = [label]
                row += [
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1-score"],
                    metrics["support"],
                ]
                report_data.append(row)

        df = pd.DataFrame(
            report_data, columns=["Class", "Precision", "Recall", "F1-score", "Support"]
        )
        df.set_index("Class", inplace=True)

        return df

    def plot_evaluation(self):
        """Plots the evaluation results including confusion matrix, ROC curve, PR curve for both classes, micro-average PR curve, and classification report."""
        cm, cm_percent = self._conf_matrix()
        roc_data = self._roc_curve_auc()
        pr_data = self._pr_curve_auc()

        fpr_1, tpr_1, roc_auc_1 = roc_data["class_1"]
        fpr_0, tpr_0, roc_auc_0 = roc_data["class_0"]

        precision_1, recall_1, pr_auc_1 = pr_data["class_1"]
        precision_0, recall_0, pr_auc_0 = pr_data["class_0"]

        df = self._class_report_df()

        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        fontsize = self.figsize[1] / 1.5

        if self.model_name is not None:
            title = f"Results for {self.model_name}"
        else:
            title = f"Evaluation Results"

        fig.suptitle(title, fontsize=fontsize * 1.5, fontweight="bold")

        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            cm,
            annot=True,
            cmap="Greens",
            fmt="d",
            linecolor="black",
            linewidths=0.7,
            ax=ax0,
            cbar=False,
        )
        ax0.set_xlabel("Predicted", fontsize=fontsize)
        ax0.set_ylabel("Actual", fontsize=fontsize)
        ax0.set_title("Confusion Matrix", fontsize=fontsize, fontweight="bold")

        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(
            df.iloc[:, :-1],
            annot=True,
            cmap="Greens",
            fmt=".2f",
            cbar=False,
            linewidths=0.7,
            linecolor="black",
            annot_kws={"weight": "bold", "fontsize": fontsize},
            ax=ax1,
        )
        ax1.set_title("Classification Report", fontsize=fontsize, fontweight="bold")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            fpr_1,
            tpr_1,
            lw=2,
            color=self.primaryColor,
            label=f"disease-gene associations ROC (AUC = {roc_auc_1:.2f})",
        )
        ax2.plot(
            fpr_0,
            tpr_0,
            lw=2,
            color=self.secondaryColor,
            label=f"non-associated genes ROC (AUC = {roc_auc_0:.2f})",
        )
        ax2.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("False Positive Rate", fontsize=fontsize)
        ax2.set_ylabel("True Positive Rate", fontsize=fontsize)
        ax2.set_title("ROC Curves", fontsize=fontsize, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=fontsize)
        ax2.grid(False)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(
            recall_1,
            precision_1,
            lw=2,
            color=self.primaryColor,
            label=f"disease-gene associations PR (AUC = {pr_auc_1:.2f})",
        )
        ax3.plot(
            recall_0,
            precision_0,
            lw=2,
            color=self.secondaryColor,
            label=f"non-associated genes PR (AUC = {pr_auc_0:.2f})",
        )
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel("Recall", fontsize=fontsize)
        ax3.set_ylabel("Precision", fontsize=fontsize)
        ax3.set_title("Precision-Recall Curves", fontsize=fontsize, fontweight="bold")
        ax3.legend(loc="lower left", fontsize=fontsize)
        ax3.grid(False)

        ax4 = fig.add_subplot(gs[2, :])

        ax4.hist(
            self.y_pred_proba[self.Y == 1],
            bins=30,
            color=self.primaryColor,
            alpha=0.5,
            density=True,
            label="disease-gene associations Probabilities",
        )
        ax4.hist(
            self.y_pred_proba[self.Y == 0],
            bins=30,
            color=self.secondaryColor,
            alpha=0.5,
            density=True,
            label="non-associated genes Probabilities",
        )

        sns.kdeplot(
            self.y_pred_proba,
            fill=True,
            color="#e5e5e5",
            ax=ax4,
            lw=2,
            label="Density",
            alpha=0.5,
        )

        ax4.set_xlabel("Probability", fontsize=fontsize)
        ax4.set_ylabel("Density", fontsize=fontsize)
        ax4.set_title(
            "Probability Distribution of Predictions",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax4.grid(False)
        ax4.legend(loc="upper right", fontsize=fontsize)

        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

    def plot_history(self):
        if self.history is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=self.history_figsize)

        axes[0].plot(
            self.history.history["loss"], label="Train Loss", color="mediumblue"
        )
        axes[0].plot(
            self.history.history["val_loss"], label="Test Loss", color="darkorange"
        )
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_ylim([0, 2])
        axes[0].legend()

        axes[1].plot(
            self.history.history["accuracy"], label="Train Accuracy", color="mediumblue"
        )
        axes[1].plot(
            self.history.history["val_accuracy"],
            label="Test Accuracy",
            color="darkorange",
        )
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)