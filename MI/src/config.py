import sys
import os

class Config:
    """Central configuration for all parameters"""

    # Data paths
    # NOTE: Update BASE_PATH if data is in a different location
    # Read environment variable
    BASE_PATH = os.environ.get("MTCAIC3_DATA_PATH")

    # Check if it's set
    if BASE_PATH is None:
        print("❌ Error: Please set the MTCAIC3_DATA_PATH environment variable before running the script.")
        print("For example: export MTCAIC3_DATA_PATH=/path/to/mtcaic3-phase-ii")
        sys.exit(1)
        
    CACHE_DIR = './' # Directory for caching processed data and model storage

    # Data parameters
    SAMPLING_RATE = 250  # Hz
    MI_TRIAL_DURATION = 9  # seconds
    MI_BASELINE_DURATION = 3.5  # seconds
    MI_STIMULATION_DURATION = 4.0  # seconds
    MI_REST_DURATION = 1.5  # seconds

    SSVEP_TRIAL_DURATION = 7  # seconds
    SSVEP_BASELINE_DURATION = 2  # seconds
    SSVEP_STIMULATION_DURATION = 4  # seconds
    SSVEP_REST_DURATION = 1  # seconds

    # EEG channels
    EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    MOTOR_CHANNELS = ['C3', 'C4']  # Motor cortex channels

    # Preprocessing parameters
    BANDPASS_LOW = 8.0  # Hz
    BANDPASS_HIGH = 30.0  # Hz
    NOTCH_FREQ = 50.0  # Hz
    NOTCH_Q = 30

    # Model training parameters
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 30
    REDUCE_LR_PATIENCE = 15

    # Augmentation parameters
    AUGMENTATION_FACTOR = 2
    NOISE_FACTOR = 0.05
    TIME_SHIFT_MAX = 25
    AMPLITUDE_SCALE_RANGE = 0.15

    # Cache files
    MI_DATA_CACHE = os.path.join(CACHE_DIR, 'mi_data_cache.pkl')
    MI_PREPROCESSED_CACHE = os.path.join(CACHE_DIR, 'mi_robust_preprocessed.pkl')
    MODEL_STORAGE_CACHE = os.path.join(CACHE_DIR, 'model_storage.pkl')


    # Model selection
    ENSEMBLE_TOP_N = 2  # Number of top models to use in ensemble

config = Config()

