import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
import xgboost as xgb
from typing import Dict, Tuple, List, Any


class SentimentClassifier:
    def __init__(self, max_features: int = 500, test_size: float = 0.25):
        """
        Initialize the sentiment classifier.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            test_size (float): Proportion of the dataset to include in the test split (0 < test_size < 1)
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1")
            
        self.max_features = max_features
        self.test_size = test_size
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_params = None
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Prepare data for training and testing.
        
        Args:
            data (pd.DataFrame): Input DataFrame with 'content' and 'sentiment' columns
            
        Returns:
            tuple: (x_train_tfidf, x_test_tfidf, y_train, y_test, labels, label_mapping)
        """
        if 'content' not in data.columns or 'sentiment' not in data.columns:
            raise ValueError("DataFrame must contain 'content' and 'sentiment' columns")

        # Check minimum samples per class
        class_counts = data['sentiment'].value_counts()
        n_classes = len(class_counts)
        min_samples_per_class = int(1 / self.test_size)  # Ensure enough samples for stratification
        
        if any(count < min_samples_per_class for count in class_counts):
            raise ValueError(
                f"Insufficient samples per class. Each class must have at least "
                f"{min_samples_per_class} samples when test_size={self.test_size}"
            )

        x = data['content'].values
        y = data['sentiment'].values
        
        # Encode labels
        y = self.label_encoder.fit_transform(y)
        labels = self.label_encoder.classes_
        label_sentiment_mapping = {
            label: sentiment 
            for label, sentiment in enumerate(labels)
        }
        
        # Split data using the configured test_size
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=self.test_size, shuffle=True, random_state=42, stratify=y
            )
        except ValueError as e:
            raise ValueError(
                f"Unable to stratify data with test_size={self.test_size}. "
                f"Ensure each class has enough samples. Error: {str(e)}"
            )
        
        # Vectorize text
        x_train_tfidf = self.tfidf_vectorizer.fit_transform(x_train).toarray()
        x_test_tfidf = self.tfidf_vectorizer.transform(x_test).toarray()
        
        return x_train_tfidf, x_test_tfidf, y_train, y_test, labels, label_sentiment_mapping
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: np.ndarray, title: str) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (np.ndarray): Label names
            title (str): Plot title
        """
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title, fontsize=20, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test: np.ndarray, y_pred_proba: np.ndarray, 
                       labels: np.ndarray, label_mapping: Dict) -> None:
        """
        Plot ROC curves for each class.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            labels (np.ndarray): Label names
            label_mapping (Dict): Mapping between encoded and original labels
        """
        n_classes = len(labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Convert y_test to one-hot encoding
        y_test_bin = np.eye(n_classes)[y_test]
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            label_name = label_mapping[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{label_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray, 
                          labels: np.ndarray, n_splits=5) -> Tuple[float, Dict[str, Any], np.ndarray]:
        """
        Train and evaluate Random Forest model with grid search.
        
        Returns:
            Tuple[float, Dict, np.ndarray]: Accuracy, best parameters, and predicted probabilities
        """
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        }
        
        clf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(clf, param_grid=params, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        
        y_pred_proba = grid.predict_proba(X_test)
        accuracy = grid.score(X_test, y_test)
        
        return accuracy, grid.best_params_, y_pred_proba
    
    def train_xgboost(self, X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray, 
                     labels: np.ndarray, n_splits=5) -> Tuple[float, Dict[str, Any], np.ndarray]:
        """
        Train and evaluate XGBoost model with grid search.
        
        Returns:
            Tuple[float, Dict, np.ndarray]: Accuracy, best parameters, and predicted probabilities
        """
        params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        clf = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(clf, param_grid=params, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        
        y_pred_proba = grid.predict_proba(X_test)
        accuracy = grid.score(X_test, y_test)
        
        return accuracy, grid.best_params_, y_pred_proba
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and probabilities
        """
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet. Call train_random_forest or train_xgboost first.")
            
        X = self.tfidf_vectorizer.transform(texts).toarray()
        y_pred = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)
        
        # Convert numeric labels back to original sentiment labels
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred_labels, y_pred_proba