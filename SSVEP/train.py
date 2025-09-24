"""
Main training script for SSVEP classification
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_loading import load_train_data, load_trial_segments
from src.preprocessing import enhanced_preprocessing
from src.augmentation import augment_trial_data
from src.feature_extraction import extract_comprehensive_features
from src.template_generation import generate_class_templates
from models.models import create_ensemble_model
from utils.training_utils import cross_validate_model, print_cv_results, save_model_artifacts

def build_enhanced_dataset(templates, use_augmentation=USE_AUGMENTATION):
    """Build comprehensive dataset using ALL subjects with augmentation."""
    print("\n🔧 Building enhanced dataset with comprehensive features...")
    print(" - Using ALL subjects for training")
    print(" - Using 4-second stimulation period only")
    print(" - Applying baseline correction with marker period")
    
    # Load data
    ssvep_train_df = load_train_data()
    
    # Extract features from ALL subjects
    X, y = [], []
    
    print(f"\n📊 Processing {len(ssvep_train_df)} SSVEP trials from all subjects...")
    
    for _, row in tqdm(ssvep_train_df.iterrows(), total=len(ssvep_train_df),
                      desc="Extracting Features"):
        try:
            # Load trial segments
            segments = load_trial_segments(row['subject_id'], row['trial_session'],
                                         row['trial'], 'train')
            
            # Data augmentation (optional)
            if use_augmentation:
                augmented_trials = augment_trial_data(segments['stimulation'])
            else:
                augmented_trials = [segments['stimulation']]
            
            for aug_trial in augmented_trials:
                features = extract_comprehensive_features(aug_trial, segments['marker'], templates)
                X.append(features)
                y.append(LABELS_MAP[row['label']])
                
        except Exception as e:
            print(f"\nError processing trial {row.get('id', 'unknown')}: {e}")
            continue
    
    print(f"\n✅ Dataset created: {len(X)} samples with {len(X[0]) if X else 0} features each")
    print(f"  - Original trials: {len(ssvep_train_df)}")
    print(f"  - After augmentation: {len(X)}")
    
    return pd.DataFrame(X), np.array(y)

def train_ensemble_model(X, y):
    """Train ensemble model with multiple algorithms."""
    print("\n🤖 Training ensemble model...")
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create ensemble model
    ensemble = create_ensemble_model()
    
    # Cross-validation evaluation
    print("\n📊 Performing 5-fold cross-validation...")
    cv_scores, cv_reports = cross_validate_model(ensemble, X_scaled, y)
    
    # Print results
    print_cv_results(cv_scores, cv_reports)
    
    # Train final model on all data
    print("\n🎯 Training final model on all data...")
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler

def main():
    """Main execution pipeline."""
    print("\n🎯 Starting Ultra-SOTA SSVEP Classification Training Pipeline...")
    print("="*80)
    
    # Generate templates
    templates = generate_class_templates()
    
    # Build dataset
    X, y = build_enhanced_dataset(templates)
    
    if len(X) == 0:
        print("❌ Error: No training data could be processed!")
        return
    
    # Train model
    model, scaler = train_ensemble_model(X, y)
    
    # Save model artifacts
    save_model_artifacts(model, scaler, templates)
    
    print("\n🎉 Training pipeline completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
