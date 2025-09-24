"""
Data loading utilities for SSVEP classification
"""
import os
import pandas as pd
import numpy as np
from src.config import *

def load_trial_segments(subject_id, session, trial, split):
    """Load trial data and return marker, stimulation, and rest segments."""
    eeg_path = os.path.join(BASE_PATH, 'SSVEP', split, subject_id, 
                           str(session), 'EEGdata.csv')
    df = pd.read_csv(eeg_path)
    
    # Calculate indices
    start_idx = (trial - 1) * TOTAL_DURATION
    marker_end = start_idx + MARKER_DURATION
    stim_end = marker_end + STIM_DURATION
    rest_end = stim_end + REST_DURATION
    
    # Extract segments
    marker_data = df.iloc[start_idx:marker_end]
    stim_data = df.iloc[marker_end:stim_end]
    rest_data = df.iloc[stim_end:rest_end]
    
    return {
        'marker': marker_data,
        'stimulation': stim_data,
        'rest': rest_data,
        'full': df.iloc[start_idx:rest_end]
    }

def load_train_data():
    """Load training data from CSV."""
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    ssvep_train_df = train_df[train_df['task'] == 'SSVEP']
    return ssvep_train_df

def load_test_data():
    """Load test data from CSV."""
    test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    return test_df

def load_golden_cohort_data():
    """Load data from golden cohort subjects only."""
    train_df = load_train_data()
    golden_train_df = train_df[train_df['subject_id'].isin(GOLDEN_SUBJECTS)]
    return golden_train_df
