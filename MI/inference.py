
# ============================================
# CELL 33: INFERENCE SCRIPT (INDEPENDENT)
# ============================================

# This script is designed to be run independently in a new environment.
# It includes necessary imports, configuration, data loading, preprocessing,
# model loading, and prediction generation.

import numpy as np
import pandas as pd
import os
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
import sys
# Scientific computing
from scipy import signal
from scipy.stats import zscore, mode

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================
# CONFIGURATION (Replicated from CELL 2)
# ============================================

class InferenceConfig:
    """Central configuration for inference"""
    # Read environment variable
    BASE_PATH = os.environ.get("MTCAIC3_DATA_PATH")

    # Check if it's set
    if BASE_PATH is None:
        print("❌ Error: Please set the MTCAIC3_DATA_PATH environment variable before running the script.")
        print("For example: export MTCAIC3_DATA_PATH=/path/to/mtcaic3-phase-ii")
        sys.exit(1)
    MODEL_PATH = './FEEGNet_Checkpoint_Attention_best.keras' # Path to the saved best model file

    # Data parameters (Replicated from training config)
    SAMPLING_RATE = 250  # Hz
    MI_TRIAL_DURATION = 9  # seconds
    MI_BASELINE_DURATION = 3.5  # seconds
    MI_STIMULATION_DURATION = 4.0  # seconds

    # EEG channels (Replicated from training config)
    # Corrected: Ensure correct EEG channels are listed. The dataset description
    # lists 'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8' (8 channels).
    # The original list had 'CZ' duplicated.
    EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']


inference_config = InferenceConfig()
print("✓ Inference Configuration loaded")


# ============================================
# DATA LOADING FUNCTION (Replicated from CELL 4 - modified for inference)
# ============================================

def convert_adc_to_uv_inference(trial_data):
    """
    Convert raw 24-bit ADC values to microvolts.
    Handles sign extension ±750 mV over 24-bit range.
    """
    data = np.asarray(trial_data)
    data24 = data.astype(np.int64, copy=False) & 0xFFFFFF

    # Perform sign-extension
    # Corrected: use 'data24' in the else part instead of 'signed'
    signed = np.where(data24 & 0x800000,
                      data24 - (1 << 24),
                      data24)

    # Scale to volts: ±0.75 V over 2^24
    volts = signed * (1.5 / (1 << 24))
    return volts * 1e6  # convert to µV

def load_test_trial_data_inference(row, base_path=None):
    """Load EEG data for a specific trial from the test set"""
    if base_path is None:
        base_path = inference_config.BASE_PATH

    # Determine dataset type based on ID range (only processing test set here)
    # For inference, we expect rows from the test.csv
    dataset = 'test'

    # Construct the path to EEGdata.csv
    eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"

    # Load the entire EEG file
    eeg_data = pd.read_csv(eeg_path)

    # Calculate indices for the specific trial
    trial_num = int(row['trial'])
    # Assuming only MI task for this inference script based on the training notebook
    samples_per_trial = int(inference_config.MI_TRIAL_DURATION * inference_config.SAMPLING_RATE)

    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial

    # Extract the trial data (only EEG channels)
    # Ensure we handle potential errors if columns are missing
    try:
        trial_data = eeg_data[inference_config.EEG_CHANNELS].iloc[start_idx:end_idx].values
    except KeyError as e:
        print(f"Error: Missing expected EEG channel in {eeg_path}. Missing channel: {e}. Available columns: {eeg_data.columns.tolist()}")
        return None # Return None if data cannot be loaded correctly
    except IndexError as e:
        print(f"Error: Index out of bounds when loading data from {eeg_path}. Start index: {start_idx}, End index: {end_idx}, Data length: {len(eeg_data)}. Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data from {eeg_path}: {e}")
        return None


    return trial_data

def load_all_test_mi_data_inference(base_path=None):
    """Load all MI test data for inference"""
    if base_path is None:
        base_path = inference_config.BASE_PATH

    print("Loading test data for inference...")
    # Load test index file
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

    # Filter for MI task only (assuming this script is for MI)
    test_mi = test_df[test_df['task'] == 'MI'].copy()

    X_test, test_ids_list = [], []
    # Use tqdm if available, otherwise a simple loop
    try:
        from tqdm import tqdm
        iterator = tqdm(test_mi.iterrows(), total=len(test_mi), desc="Loading test MI data")
    except ImportError:
        print("tqdm not found, using simple loop for progress.")
        iterator = test_mi.iterrows()


    for idx, row in iterator:
        trial_data = load_test_trial_data_inference(row, base_path)
        if trial_data is not None: # Only process if data loading was successful
            trial_uv = convert_adc_to_uv_inference(trial_data)
            X_test.append(trial_uv)
            test_ids_list.append(row['id'])
        else:
            print(f"Skipping trial with ID {{row['id']}} due to data loading error.")


    # Convert to numpy arrays
    X_test_np = np.array(X_test)
    test_ids_np = np.array(test_ids_list)

    data = {
        'X_test': X_test_np,
        'test_ids': test_ids_np,
        'test_df': test_mi # Keep the filtered dataframe for reference
    }

    return data

print("✓ Inference data loading function defined")


# ============================================
# PREPROCESSING CLASS (Replicated from CELL 6 - modified for inference)
# ============================================

class RobustMIPreprocessorInference:
    def __init__(self, sampling_rate=None):
        self.sampling_rate = sampling_rate or inference_config.SAMPLING_RATE
        self.baseline_samples = int(inference_config.MI_BASELINE_DURATION * self.sampling_rate)
        self.stimulation_samples = int(inference_config.MI_STIMULATION_DURATION * self.sampling_rate)
        self.eeg_channels = inference_config.EEG_CHANNELS
        self.motor_channels = ['C3', 'C4'] # Assuming motor channels are hardcoded

    def apply_robust_filters(self, data):
        """Apply robust filtering with error handling"""
        try:
            nyquist = self.sampling_rate / 2

            # Bandpass filter
            # Ensure filter frequencies are within Nyquist
            low_freq = 8.0
            high_freq = 30.0
            # Corrected: Use any() or all() for array comparison
            if low_freq >= nyquist or high_freq >= nyquist:
                 print(f"Warning: Filter frequencies ({{low_freq}}, {{high_freq}})...")
                 b_bp, a_bp = [1], [0] # Pass-through filter
            else:
                low, high = low_freq / nyquist, high_freq / nyquist
                b_bp, a_bp = signal.butter(4, [low, high], btype='band')

            # Notch filter
            notch_freq = 50.0
            notch_q = 30
            # Corrected: Use any() or all() for array comparison
            if notch_freq >= nyquist:
                print(f"Warning: Notch frequency ({{notch_freq}}) is >= Nyquist ({{nyquist}})...")
                b_notch, a_notch = [1], [0] # Pass-through filter
            else:
                 b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, notch_q)


            filtered_data = data.copy() # Operate on a copy

            # Corrected: Check if b_bp and a_bp are arrays before using all()
            if not (isinstance(b_bp, list) and b_bp == [1] and isinstance(a_bp, list) and a_bp == [0]): # Only apply if not pass-through
                filtered_data = signal.filtfilt(b_bp, a_bp, filtered_data, axis=0)

            # Corrected: Check if b_notch and a_notch are arrays before using all()
            if not (isinstance(b_notch, list) and b_notch == [1] and isinstance(a_notch, list) and a_notch == [0]): # Only apply if not pass-through
                filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)

            return filtered_data
        except Exception as e:
            print(f"Error during robust filtering: {{e}}. Returning original data.")
            return data # Return original data on error

    def apply_car_filtering(self, data):
        """Apply Common Average Reference with validation"""
        if data.shape[1] >= 4:
            avg_signal = np.mean(data, axis=1, keepdims=True)
            car_data = data - avg_signal

            # Validate result
            if np.any(np.isnan(car_data)) or np.any(np.isinf(car_data)):
                print("Warning: CAR filtering resulted in NaN or Inf. Returning original data.")
                return data
            return car_data
        print("Warning: Not enough channels for CAR filtering. Returning original data.")
        return data

    def robust_baseline_correction(self, stimulation_data, baseline_data):
        """Robust baseline correction with NaN handling"""
        baseline_mean = np.mean(baseline_data, axis=0, keepdims=True)
        baseline_std = np.std(baseline_data, axis=0, keepdims=True)

        # Prevent division by zero
        baseline_std = np.where(baseline_std < 1e-8, 1.0, baseline_std)

        # Z-score normalization
        corrected_data = (stimulation_data - baseline_mean) / baseline_std

        # Handle any remaining NaN or inf values
        corrected_data = np.where(np.isnan(corrected_data), 0.0, corrected_data)
        corrected_data = np.where(np.isinf(corrected_data), 0.0, corrected_data)

        # Clip extreme values
        corrected_data = np.clip(corrected_data, -10, 10)

        return corrected_data

    def extract_motor_features(self, data):
        """Extract motor cortex specific features"""
        # C3 and C4 indices - ensure they exist
        try:
            c3_idx = self.eeg_channels.index('C3')
            c4_idx = self.eeg_channels.index('C4')
        except ValueError as e:
            print(f"Error: Motor channels not found in EEG_CHANNELS: {{e}}. Returning zero features.")
            # Return zero-filled features if motor channels are missing
            return np.zeros((data.shape[0], 2))


        # Lateralization index (C3 - C4)
        lateralization = data[:, c3_idx] - data[:, c4_idx]

        # Motor cortex average
        motor_avg = (data[:, c3_idx] + data[:, c4_idx]) / 2

        # Add motor features
        motor_features = np.column_stack([lateralization, motor_avg])

        return motor_features


    def preprocess_trial(self, trial_data):
        """Preprocess a single trial data for inference"""
        try:
            # Apply robust filtering
            filtered_data = self.apply_robust_filters(trial_data)

            # Apply CAR filtering
            car_data = self.apply_car_filtering(filtered_data)

            # Extract epochs
            # Ensure data is long enough for baseline and stimulation periods
            if car_data.shape[0] < self.baseline_samples + self.stimulation_samples:
                 print(f"Warning: Trial data too short ({{car_data.shape[0]}} samples) for epoch extraction (needs at least {{self.baseline_samples + self.stimulation_samples}}). Returning truncated data.")
                 # Return truncated data or handle as an error case
                 # For now, return a zero-filled array of expected shape
                 expected_samples = int(inference_config.MI_STIMULATION_DURATION * inference_config.SAMPLING_RATE)
                 # Corrected: The combined features will have len(self.eeg_channels) + 2 channels
                 return np.zeros((expected_samples, len(self.eeg_channels) + 2))


            baseline = car_data[:self.baseline_samples]
            stimulation = car_data[self.baseline_samples:self.baseline_samples + self.stimulation_samples]

            # Robust baseline correction
            corrected_stimulation = self.robust_baseline_correction(stimulation, baseline)

            # Extract motor cortex features
            motor_features = self.extract_motor_features(corrected_stimulation)

            # Combine original channels with motor features
            # Corrected: Ensure the number of channels matches the model's expectation (8 original + 2 motor = 10)
            combined_features = np.column_stack([corrected_stimulation, motor_features])
            if combined_features.shape[1] != 10:
                 print(f"Warning: Combined features have unexpected number of channels ({{combined_features.shape[1]}}). Expected 10. Returning zero-filled data.")
                 expected_samples = int(inference_config.MI_STIMULATION_DURATION * inference_config.SAMPLING_RATE)
                 return np.zeros((expected_samples, 10))


            # Final normalization per channel
            for ch in range(combined_features.shape[1]):
                ch_data = combined_features[:, ch]
                ch_mean = np.mean(ch_data)
                ch_std = np.std(ch_data)

                if ch_std > 1e-8:
                    combined_features[:, ch] = (ch_data - ch_mean) / ch_std
                else:
                    combined_features[:, ch] = 0.0

            # Final validation
            combined_features = np.where(np.isnan(combined_features), 0.0, combined_features)
            combined_features = np.where(np.isinf(combined_features), 0.0, combined_features)
            combined_features = np.clip(combined_features, -5, 5)

            return combined_features

        except Exception as e:
            print(f"Error in preprocessing trial: {{e}}. Returning zero-filled data as fallback.")
            # Fallback to return zero-filled data of expected shape on error
            expected_samples = int(inference_config.MI_STIMULATION_DURATION * inference_config.SAMPLING_RATE)
            # The original preprocessing added 2 motor features, so 10 channels total
            return np.zeros((expected_samples, len(self.eeg_channels) + 2))


print("✓ Inference preprocessing class defined")


# ============================================
# PREPROCESSING PIPELINE (Replicated from CELL 8 - modified for inference)
# ============================================

def preprocess_test_data_inference(X_test):
    """Robust preprocessing for test data during inference"""
    print("Robust preprocessing test data for inference...")
    preprocessor = RobustMIPreprocessorInference()

    X_test_processed = []
    # Use tqdm if available, otherwise a simple loop
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(X_test)), desc="Processing test data")
    except ImportError:
        print("tqdm not found, using simple loop for progress.")
        iterator = range(len(X_test))


    for i in iterator:
        processed = preprocessor.preprocess_trial(X_test[i])
        X_test_processed.append(processed)

    # Convert to array and validate
    X_test_processed_np = np.array(X_test_processed)

    # Final validation and cleaning
    X_test_processed_np = np.where(np.isnan(X_test_processed_np), 0.0, X_test_processed_np)
    X_test_processed_np = np.where(np.isinf(X_test_processed_np), 0.0, X_test_processed_np)

    return X_test_processed_np

print("✓ Inference preprocessing pipeline defined")


# ============================================
# MODEL DEFINITIONS (Replicated from CELL 12, 13, 14)
# ============================================

# Need to redefine the model architectures as they are used in custom_objects for loading
def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aa7808
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')

    input_layer = Input(shape=(Samples, Chans, 1))

    # Layer 1: Temporal Convolution
    block1 = Conv2D(F1, (kernLength, 1), padding='same',
                    input_shape=(Samples, Chans, 1),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    # Layer 2: Depthwise Convolution
    block1 = DepthwiseConv2D((1, Chans), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(norm_rate))(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = dropoutType(dropoutRate)(block1)
    block1 = MaxPooling2D((4, 1))(block1)

    # Layer 3: Separable Convolution
    block2 = SeparableConv2D(F2, (16, 1),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = MaxPooling2D((8, 1))(block2)

    # Layer 4: Classification
    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_layer, outputs=softmax)


def EEGNet_Attention(nb_classes, Chans, Samples,
                    dropoutRate=0.5, kernLength=64, F1=8,
                    D=2, F2=16, norm_rate=0.25):
    """EEGNet with Squeeze-and-Excitation attention blocks"""
    input = Input(shape=(Samples, Chans, 1))

    # Block 1
    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(input)
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D,
                       depthwise_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=-1)(x)

    # SE Attention Block
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(1, F1*D//8), activation='relu')(se)
    se = Dense(F1*D, activation='sigmoid')(se)
    x = Multiply()([x, se])

    x = Activation('elu')(x)
    x = AveragePooling2D((4, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Block 2
    x = SeparableConv2D(F2, (16, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(x)
    output = Activation('softmax')(x)

    return Model(inputs=input, outputs=output)

def EEGNet_Attention_Improved(nb_classes, Chans, Samples,
                              dropoutRate=0.6, kernLength=64, F1=16,
                              D=2, F2=32, norm_rate=0.25):
    """Improved EEGNet with Squeeze-and-Excitation attention blocks and minor architectural changes"""
    input = Input(shape=(Samples, Chans, 1))

    # Block 1
    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False,
               kernel_regularizer=l2(0.001))(input) # Added L2 regularization
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D,
                       depthwise_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=-1)(x)

    # SE Attention Block
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(1, F1*D//8), activation='relu')(se)
    se = Dense(F1*D, activation='sigmoid')(se)
    x = Multiply()([x, se])

    x = Activation('elu')(x)
    x = AveragePooling2D((4, 1))(x)
    x = Dropout(dropoutRate)(x) # Increased dropout rate

    # Block 2
    x = SeparableConv2D(F2, (16, 1), use_bias=False, padding='same',
                        pointwise_regularizer=l2(0.001))(x) # Corrected to pointwise_regularizer
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropoutRate)(x) # Increased dropout rate

    # Added Block 3 (another Separable Convolutional Block)
    x = SeparableConv2D(F2 * 2, (16, 1), use_bias=False, padding='same', # Increased filters
                        pointwise_regularizer=l2(0.001))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((2, 1))(x) # Smaller pooling to retain more information
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate),
              kernel_regularizer=l2(0.001))(x) # Added L2 regularization
    output = Activation('softmax')(x)

    return Model(inputs=input, outputs=output)


print("✓ Model definitions loaded")

# ============================================
# INFERENCE SCRIPT - MAIN EXECUTION
# ============================================

# Load test data
test_data = load_all_test_mi_data_inference(base_path=inference_config.BASE_PATH)
X_test_raw = test_data['X_test']
test_ids = test_data['test_ids']

# Apply preprocessing to test data
X_test_proc = preprocess_test_data_inference(X_test_raw)

# Reshape processed test data for model input (assuming 4D input: samples, time, channels, 1)
# The preprocessing function should output data with shape (num_samples, time_steps, channels)
# Model expects (num_samples, time_steps, channels, 1)
# Determine samples and channels from the processed data shape
if X_test_proc.ndim == 3:
    samples, time_steps, chans = X_test_proc.shape
    X_test_eeg = np.expand_dims(X_test_proc, axis=-1)
    print(f"Processed test data reshaped to: {{X_test_eeg.shape}}")
else:
    print(f"Error: Unexpected processed test data shape: {{X_test_proc.shape}}. Expected 3 dimensions.")
    # Handle error - perhaps exit or raise exception
    X_test_eeg = None # Set to None to prevent further execution if shape is wrong


if X_test_eeg is not None:
    # Load the best model
    model_path = inference_config.MODEL_PATH

    print(f"\nLoading the best model from: {{model_path}}")

    # Ensure custom objects are available if needed (e.g., Attention layers)
    # Custom objects dictionary should include all custom layers used in the saved model
    custom_objects_inference = {
        'EEGNet_Attention': EEGNet_Attention,
        'EEGNet_Attention_Improved': EEGNet_Attention_Improved,
        'Multiply': Multiply, # Add Multiply if it's used in a custom way
        'DepthwiseConv2D': DepthwiseConv2D, # Add if DepthwiseConv2D is used in a custom way
        'SeparableConv2D': SeparableConv2D, # Add if SeparableConv2D is used in a custom way
        'GlobalAveragePooling2D': GlobalAveragePooling2D, # Add if used
        'AveragePooling2D': AveragePooling2D, # Add if used
        'max_norm': max_norm, # Add if max_norm constraint is used
        'l2': l2 # Add if l2 regularizer is used
    }


    # Load the model with custom objects
    try:
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects_inference)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {{e}}")
        print("Please ensure the MODEL_PATH in the InferenceConfig is correct and that all custom layers are defined.")
        best_model = None # Set to None if loading fails

    if best_model:
        # Generate predictions on the test set
        print("\nGenerating predictions on the test set...")
        test_predictions_prob = best_model.predict(X_test_eeg)
        test_predictions_classes = np.argmax(test_predictions_prob, axis=1)
        print("✓ Predictions generated")

        # Convert predictions to labels (0 -> Left, 1 -> Right)
        predicted_labels = ['Left' if pred == 0 else 'Right' for pred in test_predictions_classes]

        # Create the submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_ids,
            'label': predicted_labels
        })

        # Save the submission file
        submission_filename = 'inference_submission.csv' # Use a fixed name for the inference script output
        submission_df.to_csv(submission_filename, index=False)

        print(f"\n✓ Inference submission saved to '{{submission_filename}}'")
        print(f"Prediction distribution:")
        print(submission_df['label'].value_counts())

    else:
        print("\nSkipping prediction and submission creation due to model loading failure.")
else:
    print("\nSkipping prediction and submission creation due to processed data shape issue.")

