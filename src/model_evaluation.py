from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




class ModelEvaluation:
    def __init__(self, model, X_test, y_test, model_name):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

    def _conf_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(len(cm)):
            if total_actual[i] > 0:
                cm_percent[i, i] = true_positives[i] / total_actual[i]
                
        return cm, cm_percent

    def _roc_curve_auc(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return fpr, tpr, roc_auc
    
    def _class_report_df(self):
        class_report = classification_report(self.y_test, self.y_pred, output_dict=True)
        report_data = []
        for label, metrics in class_report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                row = [label]
                row += [metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                report_data.append(row)
                
        df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        df.set_index('Class', inplace=True)
        
        return df

    def plot_evaluation(self):
        """Plots the evaluation results including confusion matrix, ROC curve, and classification report."""
        cm, cm_percent = self._conf_matrix()
        fpr, tpr, roc_auc = self._roc_curve_auc()
        df = self._class_report_df()
        
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f'Results for {self.model_name}', fontsize=10, fontweight='bold')
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linecolor='black', linewidths=.7, ax=ax[0, 0], cbar=False)
        
        ax[0, 0].set_xlabel('Predicted')
        ax[0, 0].set_ylabel('Actual')

        ax[0, 1].plot(fpr, tpr, lw=2, color='#1f77b4', label=f'ROC curve (area = {roc_auc:.2f})')
        ax[0, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax[0, 1].set_xlim([0.0, 1.0])
        ax[0, 1].set_ylim([0.0, 1.05])
        ax[0, 1].set_xlabel('False Positive Rate')
        ax[0, 1].set_ylabel('True Positive Rate')
        ax[0, 1].legend(loc='lower right')
        ax[0, 1].grid(False)

        ax[1, 0].hist(self.y_pred_proba, bins=30, color='green', alpha=0.7)
        ax[1, 0].set_xlabel('Probability')
        ax[1, 0].set_ylabel('Frequency')
        ax[1, 0].grid(False)

        sns.heatmap(df.iloc[:, :-1], 
                    annot=True, 
                    cmap='Blues',  
                    fmt=".2f", 
                    cbar=False, 
                    linewidths=.7, 
                    linecolor='black', 
                    annot_kws={"weight": "bold", "fontsize": 10}, 
                    ax=ax[1, 1])

        plt.tight_layout()
        plt.show()
    
    
    
class TfEvaluation:
    def __init__(self, model, X_test, y_test, model_name, history=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.history = history
        self.y_pred_proba = self.model.predict(self.X_test).flatten()
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)

    def _roc_curve_auc(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def _conf_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(len(cm)):
            if total_actual[i] > 0:
                cm_percent[i, i] = true_positives[i] / total_actual[i]

        return cm, cm_percent

    def _class_report_df(self):
        class_report = classification_report(self.y_test, self.y_pred, output_dict=True)
        report_data = []
        for label, metrics in class_report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                row = [label]
                row += [metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                report_data.append(row)

        df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        df.set_index('Class', inplace=True)
        
        return df

    def plot_evaluation(self):
        cm, cm_percent = self._conf_matrix()
        fpr, tpr, roc_auc = self._roc_curve_auc()
        df = self._class_report_df()

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f'Results for {self.model_name}', fontsize=14, fontweight='bold')

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linecolor='black', linewidths=.7, ax=ax[0, 0], cbar=False)
        ax[0, 0].set_xlabel('Predicted')
        ax[0, 0].set_ylabel('Actual')
        ax[0, 0].set_title('Confusion Matrix')

        ax[0, 1].plot(fpr, tpr, lw=2, color='#1f77b4', label=f'ROC curve (area = {roc_auc:.2f})')
        ax[0, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax[0, 1].set_xlim([0.0, 1.0])
        ax[0, 1].set_ylim([0.0, 1.05])
        ax[0, 1].set_xlabel('False Positive Rate')
        ax[0, 1].set_ylabel('True Positive Rate')
        ax[0, 1].legend(loc='lower right')
        ax[0, 1].set_title('ROC Curve')
        ax[0, 1].grid(False)

        ax[1, 0].hist(self.y_pred_proba, bins=30, color='green', alpha=0.7)
        ax[1, 0].set_xlabel('Probability')
        ax[1, 0].set_ylabel('Frequency')
        ax[1, 0].set_title('Prediction Probability Distribution')
        ax[1, 0].grid(False)

        sns.heatmap(df.iloc[:, :-1], 
                   annot=True, 
                   cmap='Blues', 
                   fmt=".2f", 
                   cbar=False, 
                   linewidths=.7, 
                   linecolor='black', 
                   annot_kws={"weight": "bold", "fontsize": 10}, 
                   ax=ax[1, 1])
        ax[1, 1].set_title('Classification Report')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_history(self):
        if self.history is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(8, 5))

        axes[0].plot(self.history.history["loss"], label="Train Loss", color='mediumblue')
        axes[0].plot(self.history.history["val_loss"], label="Test Loss", color='darkorange')
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_ylim([0, 2])
        axes[0].legend()

        axes[1].plot(self.history.history["accuracy"], label="Train Accuracy", color='mediumblue')
        axes[1].plot(self.history.history["val_accuracy"], label="Test Accuracy", color='darkorange')
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        plt.tight_layout()
        plt.show()