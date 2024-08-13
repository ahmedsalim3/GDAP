from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


class ModelEvaluation:
    def __init__(self, y_test, y_pred, model):
        self.y_test = y_test
        self.y_pred = y_pred
        self.model = model

    def conf_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)

        for i in range(len(cm)):
            cm_percent[i, i] = true_positives[i] / total_actual[i]

        return cm, cm_percent

    def roc_curve(self, clf, X_test):
        y_score = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def class_report(self, clf, X_test):
        y_pred = clf.predict(X_test)
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        report_data = []
        for label, metrics in class_report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                row = [label]
                row += [metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                report_data.append(row)
        
        df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        df.set_index('Class', inplace=True)
        
        return df

    def plot_results(self, clf, X_test):
        cm, cm_percent = self.conf_matrix()
        fpr, tpr, roc_auc = self.roc_curve(clf, X_test)
        df = self.class_report(clf, X_test)

        probabilities = clf.predict_proba(X_test)[:, 1]

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linecolor='black', linewidths=.7, ax=ax[0, 0])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j:
                    ax[0, 0].text(j + 0.5, i + 0.8, f'{cm_percent[i, j] * 100:.2f}%', ha='center', va='center', color='white')
        ax[0, 0].set_xlabel('Predicted')
        ax[0, 0].set_ylabel('Actual')
        ax[0, 0].set_title(f'Confusion Matrix\nModel: {self.model}')
        
        # ROC Curve
        ax[0, 1].plot(fpr, tpr, lw=2, color='#1f77b4', label=f'ROC curve (area = {roc_auc:.2f})')
        ax[0, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax[0, 1].set_xlim([0.0, 1.0])
        ax[0, 1].set_ylim([0.0, 1.05])
        ax[0, 1].set_xlabel('False Positive Rate')
        ax[0, 1].set_ylabel('True Positive Rate')
        ax[0, 1].set_title(f'ROC Curve\nModel: {self.model}')
        ax[0, 1].legend(loc='lower right')

        # Probabilitied
        ax[1, 0].hist(probabilities, bins=30, color='green', alpha=0.7)
        ax[1, 0].set_xlabel('Probability')
        ax[1, 0].set_ylabel('Frequency')
        ax[1, 0].set_title(f'Predicted Probabilities Distribution\nModel: {self.model}')
        
        # Class Report
        sns.heatmap(df.iloc[:, :-1], annot=True, cmap='Blues', fmt=".2f", cbar=False, linecolor='black', linewidths=.7, annot_kws={"weight": "bold", "fontsize": 10}, ax=ax[1, 1], vmin=0, vmax=1)
        ax[1, 1].set_title(f'Classification Report\nModel: {self.model}')

        plt.tight_layout()
        plt.show()
        