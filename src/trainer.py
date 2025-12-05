"""Model training for casualty severity classification"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import joblib
from pathlib import Path


class SeverityClassifier:
    """Train models to predict casualty severity (Fatal/Serious/Slight)"""

    def __init__(self, model_type='mlp'):
        """
        Initialize classifier

        Args:
            model_type: 'lr', 'rf', 'mlp', or 'xgb'
        """
        self.model_type = model_type
        self.model = self._build_model(model_type)

    def _build_model(self, model_type):
        """Build model pipeline based on type"""

        if model_type == 'lr':
            # Logistic Regression with scaling and balanced class weights
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ))
            ])

        elif model_type == 'rf':
            # Random Forest with balanced class weights
            return Pipeline([
                ('clf', RandomForestClassifier(
                    class_weight='balanced',
                    n_estimators=100,
                    random_state=42
                ))
            ])

        elif model_type == 'mlp':
            # MLP with SMOTE for imbalanced data
            return ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', MLPClassifier(
                    hidden_layer_sizes=(50,),
                    activation='relu',
                    alpha=0.001,
                    max_iter=500,
                    random_state=42
                ))
            ])

        elif model_type == 'xgb':
            # XGBoost with scale_pos_weight for imbalance
            return Pipeline([
                ('clf', XGBClassifier(
                    scale_pos_weight=3,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=42
                ))
            ])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X, y):
        """
        Train model using Stratified K-Fold CV

        Args:
            X: Feature matrix
            y: Target vector (0=Fatal, 1=Serious, 2=Slight)

        Returns:
            dict: Performance metrics
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Dataset size: {len(X)} samples")
        print(f"Class distribution: {np.bincount(y)}")

        # Stratified K-Fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validation scores
        f1_scores = cross_val_score(self.model, X, y, cv=cv,
                                     scoring='f1_macro', n_jobs=-1)
        recall_scores = cross_val_score(self.model, X, y, cv=cv,
                                         scoring='recall_macro', n_jobs=-1)

        # Train on full dataset for final model
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        metrics = {
            'f1_cv_mean': f1_scores.mean(),
            'f1_cv_std': f1_scores.std(),
            'recall_cv_mean': recall_scores.mean(),
            'recall_cv_std': recall_scores.std(),
            'f1_train': f1_score(y, y_pred, average='macro'),
            'recall_train': recall_score(y, y_pred, average='macro'),
            'precision_train': precision_score(y, y_pred, average='macro')
        }

        print(f"\nCross-Validation Results:")
        print(f"  F1-score (macro): {metrics['f1_cv_mean']:.3f} ± {metrics['f1_cv_std']:.3f}")
        print(f"  Recall (macro):   {metrics['recall_cv_mean']:.3f} ± {metrics['recall_cv_std']:.3f}")

        print(f"\nTraining Set Performance:")
        print(f"  F1-score:   {metrics['f1_train']:.3f}")
        print(f"  Recall:     {metrics['recall_train']:.3f}")
        print(f"  Precision:  {metrics['precision_train']:.3f}")

        return metrics

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

    def save_model(self, filepath='models/severity_model.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type
        }, filepath)
        print(f"\nModel saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        classifier = SeverityClassifier(model_type=data['model_type'])
        classifier.model = data['model']
        return classifier


def compare_models(X, y):
    """
    Compare all models and return results

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        dict: Results for all models
    """
    models = {
        'Logistic Regression': 'lr',
        'Random Forest': 'rf',
        'MLP': 'mlp',
        'XGBoost': 'xgb'
    }

    results = {}

    for name, model_type in models.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        classifier = SeverityClassifier(model_type=model_type)
        metrics = classifier.train(X, y)
        results[name] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - Cross-Validation F1 Scores (Macro)")
    print(f"{'='*60}")
    for name, metrics in results.items():
        print(f"{name:20s}: {metrics['f1_cv_mean']:.3f}")

    return results
