"""
Training utilities for model evaluation and saving
"""
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from src.config import *

def cross_validate_model(model, X, y, n_splits=CROSS_VALIDATION_FOLDS):
    """Perform cross-validation and return scores."""
    cv_scores = []
    cv_reports = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_val_cv)
        
        accuracy = accuracy_score(y_val_cv, y_pred_cv)
        cv_scores.append(accuracy)
        
        # Generate classification report
        report = classification_report(y_val_cv, y_pred_cv,
                                     target_names=list(SSVEP_FREQS.keys()),
                                     output_dict=True)
        cv_reports.append(report)
        
        print(f"\nFold {fold} Accuracy: {accuracy:.4f}")
    
    return cv_scores, cv_reports

def print_cv_results(cv_scores, cv_reports):
    """Print detailed cross-validation results."""
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"Average Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Print detailed classification report (average across folds)
    print("\nAverage Classification Report:")
    print("-"*60)
    
    # Calculate average metrics
    avg_report = {}
    for label in list(SSVEP_FREQS.keys()) + ['accuracy', 'macro avg', 'weighted avg']:
        if label == 'accuracy':
            avg_report[label] = np.mean(cv_scores)
        else:
            avg_report[label] = {}
            for metric in ['precision', 'recall', 'f1-score']:
                if label in ['macro avg', 'weighted avg']:
                    values = [r[label][metric] for r in cv_reports if label in r]
                else:
                    values = [r[label][metric] for r in cv_reports if label in r]
                if values:
                    avg_report[label][metric] = np.mean(values)
    
    # Print formatted report
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*45)
    for label in SSVEP_FREQS.keys():
        if label in avg_report:
            print(f"{label:<15} {avg_report[label]['precision']:<10.3f} "
                  f"{avg_report[label]['recall']:<10.3f} "
                  f"{avg_report[label]['f1-score']:<10.3f}")
    print("-"*45)
    print(f"{'Macro Avg':<15} {avg_report['macro avg']['precision']:<10.3f} "
          f"{avg_report['macro avg']['recall']:<10.3f} "
          f"{avg_report['macro avg']['f1-score']:<10.3f}")
    print(f"{'Weighted Avg':<15} {avg_report['weighted avg']['precision']:<10.3f} "
          f"{avg_report['weighted avg']['recall']:<10.3f} "
          f"{avg_report['weighted avg']['f1-score']:<10.3f}")
    print("="*80)

def save_model_artifacts(model, scaler, templates):
    """Save model, scaler, and templates to checkpoint directory."""
    print("\n💾 Saving model artifacts...")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Save artifacts
    joblib.dump(model, MODEL_CHECKPOINT)
    joblib.dump(scaler, SCALER_CHECKPOINT)
    joblib.dump(templates, TEMPLATES_CHECKPOINT)
    
    print(f"✅ Model saved to: {MODEL_CHECKPOINT}")
    print(f"✅ Scaler saved to: {SCALER_CHECKPOINT}")
    print(f"✅ Templates saved to: {TEMPLATES_CHECKPOINT}")

def load_model_artifacts():
    """Load model, scaler, and templates from checkpoint directory."""
    print("\n📂 Loading model artifacts...")
    
    model = joblib.load(MODEL_CHECKPOINT)
    scaler = joblib.load(SCALER_CHECKPOINT)
    templates = joblib.load(TEMPLATES_CHECKPOINT)
    
    print("✅ Model artifacts loaded successfully")
    
    return model, scaler, templates
