"""
Inference script for SSVEP classification
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_loading import load_test_data, load_trial_segments
from src.feature_extraction import extract_comprehensive_features
from utils.training_utils import load_model_artifacts

def generate_predictions(model, scaler, templates):
    """Generate predictions for test data using stimulation period only."""
    print("\n🎯 Generating predictions for test set...")
    
    test_df = load_test_data()
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        if row['task'] == 'SSVEP':
            try:
                # Load trial segments
                segments = load_trial_segments(row['subject_id'], row['trial_session'],
                                             row['trial'], 'test')
                
                # Extract features from stimulation period with baseline correction
                features = extract_comprehensive_features(segments['stimulation'], 
                                                        segments['marker'], 
                                                        templates)
                
                # Convert to DataFrame and scale
                feature_df = pd.DataFrame([features]).fillna(0)
                feature_scaled = scaler.transform(feature_df)
                
                # Predict
                pred_numeric = model.predict(feature_scaled)[0]
                pred_label = REVERSE_LABELS_MAP[pred_numeric]
                predictions.append(pred_label)
                
            except Exception as e:
                print(f"\nError predicting trial {row['id']}: {e}")
                predictions.append('Forward')  # Default
        else:
            # MI task placeholder
            predictions.append('Left')
    
    return predictions

def main():
    """Main execution pipeline for inference."""
    print("\n🎯 Starting SSVEP Classification Inference Pipeline...")
    print("="*80)
    
    # Load model artifacts
    model, scaler, templates = load_model_artifacts()
    
    # Generate predictions
    predictions = generate_predictions(model, scaler, templates)
    
    # Create submission
    test_df = load_test_data()
    submission = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print("\n✅ Submission file created: submission.csv")
    print("\n📊 Prediction Summary:")
    print(submission['label'].value_counts())
    print(f"\n🎉 Inference pipeline completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
