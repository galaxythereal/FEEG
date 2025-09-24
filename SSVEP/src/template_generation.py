"""
Template generation utilities for SSVEP classification
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import *
from src.data_loading import load_trial_segments, load_golden_cohort_data
from src.preprocessing import enhanced_preprocessing

def generate_class_templates():
    """Generate enhanced class templates using golden subjects and stimulation period only."""
    print("\n📊 Generating class templates from Golden Cohort...")
    
    golden_train_df = load_golden_cohort_data()
    templates = {label: [] for label in SSVEP_FREQS.keys()}
    
    for label in tqdm(SSVEP_FREQS.keys(), desc="Building Templates"):
        label_trials = golden_train_df[golden_train_df['label'] == label]
        
        for _, row in label_trials.head(40).iterrows():
            try:
                # Load trial segments
                segments = load_trial_segments(row['subject_id'], row['trial_session'],
                                             row['trial'], 'train')
                
                # Preprocess stimulation data with baseline correction
                eeg_clean = enhanced_preprocessing(segments['stimulation'], 
                                                 segments['marker'],
                                                 use_ica=True, 
                                                 use_laplacian=False)
                templates[label].append(eeg_clean.T.values)
            except:
                continue
    
    # Average templates
    final_templates = {}
    for label, trials in templates.items():
        if trials:
            final_templates[label] = np.mean(np.stack(trials), axis=0)
            print(f"  - {label}: {len(trials)} trials averaged")
        else:
            # Fallback: create synthetic template
            final_templates[label] = np.random.randn(len(ALL_EEG_CHANNELS), STIM_DURATION) * 0.1
            print(f"  - {label}: Using synthetic template (no data)")
    
    return final_templates
