"""
Model definitions for SSVEP classification
"""
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.config import MODEL_PARAMS

def create_ensemble_model():
    """Create ensemble model with multiple algorithms."""
    # Define base models
    xgb_model = xgb.XGBClassifier(**MODEL_PARAMS['xgb'])
    svm_model = SVC(**MODEL_PARAMS['svm'])
    lr_model = LogisticRegression(**MODEL_PARAMS['lr'])
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('svm', svm_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    return ensemble
