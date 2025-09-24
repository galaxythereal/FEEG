"""
Central configuration for SSVEP classification project
"""
import os
import sys

# Base path configuration
BASE_PATH = os.environ.get("MTCAIC3_DATA_PATH")
if BASE_PATH is None:
    print("❌ Error: Please set the MTCAIC3_DATA_PATH environment variable before running the script.")
    print("For example: export MTCAIC3_DATA_PATH=/path/to/mtcaic3-phase-ii")
    sys.exit(1)

# Sampling and signal parameters
SAMPLING_RATE = 250

# Channel definitions
ALL_EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
MOTION_CHANNELS = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']

# SSVEP frequencies and labels
SSVEP_FREQS = {'Forward': 7, 'Backward': 8, 'Left': 10, 'Right': 13}
LABELS_MAP = {label: i for i, label in enumerate(SSVEP_FREQS.keys())}
REVERSE_LABELS_MAP = {i: label for label, i in LABELS_MAP.items()}

# Trial timing parameters
MARKER_DURATION = 2 * SAMPLING_RATE  # 500 samples (2s preparation)
STIM_DURATION = 4 * SAMPLING_RATE    # 1000 samples (4s stimulation)
REST_DURATION = 1 * SAMPLING_RATE     # 250 samples (1s rest)
TOTAL_DURATION = MARKER_DURATION + STIM_DURATION + REST_DURATION  # 1750 samples

# Golden Cohort for template generation
GOLDEN_SUBJECTS = ['S2', 'S4', 'S7', 'S9', 'S10', 'S12', 'S16', 'S17', 
                   'S20', 'S21', 'S24', 'S25', 'S29', 'S30']

# Frequency bands for Filter Bank analysis
FILTER_BANKS = [
    (6, 14),   # Primary SSVEP band
    (14, 22),  # Second harmonic
    (22, 30),  # Third harmonic
    (30, 40),  # High frequency
    (8, 30)    # Broad band
]

# Model parameters
MODEL_PARAMS = {
    'xgb': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    },
    'svm': {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    },
    'lr': {
        'C': 1.0,
        'random_state': 42,
        'max_iter': 1000
    }
}

# Training parameters
CROSS_VALIDATION_FOLDS = 5
USE_AUGMENTATION = True

# File paths
CHECKPOINT_DIR = 'checkpoints'
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'ensemble_model.joblib')
SCALER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'scaler.joblib')
TEMPLATES_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'templates.joblib')
