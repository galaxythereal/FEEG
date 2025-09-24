
import os
import numpy as np
from scipy import signal
from scipy.stats import zscore
from tqdm import tqdm
import pickle
from .config import config # Corrected import for package structure

class RobustMIPreprocessor:
    """
    Robust preprocessing pipeline for Motor Imagery EEG data.

    Includes filtering, CAR, baseline correction, feature extraction,
    and robust handling of potential errors and data inconsistencies.
    """
    def __init__(self, sampling_rate=None):
        self.sampling_rate = sampling_rate or config.SAMPLING_RATE
        # Calculate samples based on configuration and sampling rate
        self.baseline_samples = int(config.MI_BASELINE_DURATION * self.sampling_rate)
        self.stimulation_samples = int(config.MI_STIMULATION_DURATION * self.sampling_rate)
        self.eeg_channels = config.EEG_CHANNELS
        self.motor_channels = config.MOTOR_CHANNELS # Use motor channels from config

    def apply_robust_filters(self, data):
        """Apply robust filtering (bandpass and notch) with error handling."""
        try:
            nyquist = self.sampling_rate / 2

            # Bandpass filter
            low_freq = config.BANDPASS_LOW
            high_freq = config.BANDPASS_HIGH
            # Check if filter frequencies are within Nyquist
            if low_freq >= nyquist or high_freq >= nyquist:
                 print(f"Warning: Bandpass frequencies ({{low_freq}}, {{high_freq}}) are >= Nyquist ({{nyquist}}). Skipping bandpass filter.")
                 b_bp, a_bp = [1], [0] # Pass-through filter
            else:
                low, high = low_freq / nyquist, high_freq / nyquist
                # Corrected: Ensure filter coefficients are valid
                try:
                    b_bp, a_bp = signal.butter(4, [low, high], btype='band')
                except ValueError as e:
                    print(f"Error creating bandpass filter: {{e}}. Skipping bandpass filter.")
                    b_bp, a_bp = [1], [0] # Pass-through filter


            # Notch filter
            notch_freq = config.NOTCH_FREQ
            notch_q = config.NOTCH_Q
            # Check if notch frequency is within Nyquist
            if notch_freq >= nyquist:
                 print(f"Warning: Notch frequency ({{notch_freq}}) is >= Nyquist ({{nyquist}}). Skipping notch filter.")
                 b_notch, a_notch = [1], [0] # Pass-through filter
            else:
                 # Corrected: Ensure filter coefficients are valid
                 try:
                    b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, notch_q)
                 except ValueError as e:
                     print(f"Error creating notch filter: {{e}}. Skipping notch filter.")
                     b_notch, a_notch = [1], [0] # Pass-through filter


            filtered_data = data.copy() # Operate on a copy to avoid modifying original data

            # Apply bandpass filter if not pass-through
            # Corrected: Check if b_bp and a_bp are lists before checking for pass-through
            if not (isinstance(b_bp, list) and b_bp == [1] and isinstance(a_bp, list) and a_bp == [0]):
                 # Corrected: Ensure data is not empty before filtering
                 if filtered_data.shape[0] > 0:
                    filtered_data = signal.filtfilt(b_bp, a_bp, filtered_data, axis=0)
                 else:
                     print("Warning: Data is empty, skipping bandpass filter.")


            # Apply notch filter if not pass-through
            # Corrected: Check if b_notch and a_notch are lists before checking for pass-through
            if not (isinstance(b_notch, list) and b_notch == [1] and isinstance(a_notch, list) and a_notch == [0]):
                 # Corrected: Ensure data is not empty before filtering
                 if filtered_data.shape[0] > 0:
                    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)
                 else:
                     print("Warning: Data is empty, skipping notch filter.")


            return filtered_data
        except Exception as e:
            print(f"Error during robust filtering: {{e}}. Returning original data.")
            return data # Return original data on error


    def apply_car_filtering(self, data):
        """Apply Common Average Reference (CAR) with validation."""
        # Corrected: Ensure data is not empty before applying CAR
        if data.shape[0] == 0:
             print("Warning: Data is empty, skipping CAR filtering.")
             return data

        if data.shape[1] >= 4: # CAR is meaningful with at least 4 channels
            avg_signal = np.mean(data, axis=1, keepdims=True)
            car_data = data - avg_signal

            # Validate result
            if np.any(np.isnan(car_data)) or np.any(np.isinf(car_data)):
                print("Warning: CAR filtering resulted in NaN or Inf. Returning original data.")
                return data
            return car_data
        print("Warning: Not enough channels ({{data.shape[1]}}) for CAR filtering. Returning original data.")
        return data

    def robust_baseline_correction(self, stimulation_data, baseline_data):
        """Apply robust baseline correction using mean and std of baseline data."""
        # Corrected: Handle empty baseline or stimulation data
        if baseline_data.shape[0] == 0 or stimulation_data.shape[0] == 0:
             print("Warning: Empty baseline or stimulation data for baseline correction. Returning stimulation data without correction.")
             return stimulation_data


        baseline_mean = np.mean(baseline_data, axis=0, keepdims=True)
        baseline_std = np.std(baseline_data, axis=0, keepdims=True)

        # Prevent division by zero
        baseline_std = np.where(baseline_std < 1e-8, 1.0, baseline_std)

        # Z-score normalization
        corrected_data = (stimulation_data - baseline_mean) / baseline_std

        # Handle any remaining NaN or inf values introduced by operations
        corrected_data = np.where(np.isnan(corrected_data), 0.0, corrected_data)
        corrected_data = np.where(np.isinf(corrected_data), 0.0, corrected_data)

        # Clip extreme values to prevent outliers from dominating
        corrected_data = np.clip(corrected_data, -10, 10)

        return corrected_data

    def extract_motor_features(self, data):
        """Extract motor cortex specific features (Lateralization Index and Motor Average)."""
        # Corrected: Handle empty data
        if data.shape[0] == 0:
             print("Warning: Data is empty for motor feature extraction. Returning zero features.")
             return np.zeros((0, 2))

        # C3 and C4 indices - ensure they exist in the current channels
        try:
            # This assumes the order of channels in 'data' is consistent with 'self.eeg_channels'
            # A more robust approach would explicitly pass channel names with the data or
            # ensure the data loading function guarantees channel order.
            # For now, we rely on the assumption from the original notebook structure.
            c3_idx = self.eeg_channels.index('C3')
            c4_idx = self.eeg_channels.index('C4')

            # Ensure indices are within the bounds of the data
            if c3_idx >= data.shape[1] or c4_idx >= data.shape[1]:
                 print(f"Error: Motor channel index out of bounds. Data has {{data.shape[1]}} channels. C3 index: {{c3_idx}}, C4 index: {{c4_idx}}. Returning zero features.")
                 return np.zeros((data.shape[0], 2))

        except ValueError as e:
            print(f"Error: Motor channels 'C3' or 'C4' not found in the expected EEG_CHANNELS list: {{e}}. Returning zero features.")
            # Return zero-filled features if motor channels are missing
            return np.zeros((data.shape[0], 2))


        # Lateralization index (C3 - C4)
        lateralization = data[:, c3_idx] - data[:, c4_idx]

        # Motor cortex average
        motor_avg = (data[:, c3_idx] + data[:, c4_idx]) / 2

        # Combine as a new set of features
        motor_features = np.column_stack([lateralization, motor_avg])

        return motor_features

    def preprocess_trial(self, trial_data):
        """Apply the full preprocessing pipeline to a single trial's raw data."""
        # Corrected: Handle empty input data
        if trial_data is None or trial_data.shape[0] == 0:
             print("Warning: Input trial data is empty or None. Returning zero-filled data of expected shape.")
             expected_samples = int(config.MI_STIMULATION_DURATION * config.SAMPLING_RATE)
             expected_channels = len(self.eeg_channels) + 2 # Original channels + 2 motor features
             return np.zeros((expected_samples, expected_channels))


        try:
            # Apply robust filtering
            filtered_data = self.apply_robust_filters(trial_data)

            # Apply CAR filtering
            car_data = self.apply_car_filtering(filtered_data)

            # Extract epochs
            expected_total_samples = self.baseline_samples + self.stimulation_samples
            # Corrected: Handle cases where data is too short for full epoch extraction
            if car_data.shape[0] < expected_total_samples:
                 print(f"Warning: Trial data too short ({{car_data.shape[0]}} samples) for full epoch extraction (needs at least {{expected_total_samples}}). Processing available data.")
                 # Use available data for baseline and stimulation, padding if necessary
                 baseline = car_data[:min(self.baseline_samples, car_data.shape[0])]
                 stimulation = car_data[min(self.baseline_samples, car_data.shape[0]):min(expected_total_samples, car_data.shape[0])]

                 # Pad stimulation if it's shorter than expected
                 if stimulation.shape[0] < self.stimulation_samples:
                     padding_needed = self.stimulation_samples - stimulation.shape[0]
                     stimulation = np.pad(stimulation, ((0, padding_needed), (0, 0)), mode='constant') # Pad with zeros

                 # If baseline is completely missing, cannot do robust baseline correction
                 if baseline.shape[0] == 0:
                     print("Warning: Baseline data is missing. Skipping robust baseline correction.")
                     corrected_stimulation = stimulation # Use stimulation data directly
                 else:
                     # Attempt robust baseline correction with potentially shorter baseline
                     corrected_stimulation = self.robust_baseline_correction(stimulation, baseline)

            else: # Data is long enough for full epoch extraction
                 baseline = car_data[:self.baseline_samples]
                 stimulation = car_data[self.baseline_samples:expected_total_samples]
                 # Apply robust baseline correction
                 corrected_stimulation = self.robust_baseline_correction(stimulation, baseline)


            # Extract motor cortex features
            motor_features = self.extract_motor_features(corrected_stimulation)

            # Combine original channels with motor features
            # Ensure the number of channels matches the model's expectation (8 original + 2 motor = 10)
            expected_total_channels = len(self.eeg_channels) + len(self.motor_channels) # Expect 10 channels
            combined_features = np.column_stack([corrected_stimulation, motor_features])

            # Corrected: Ensure the shape matches the expected stimulation samples and total channels
            # Pad the combined features if the stimulation epoch was shorter than expected initially
            if combined_features.shape[0] != self.stimulation_samples or combined_features.shape[1] != expected_total_channels:
                 print(f"Warning: Preprocessed features have unexpected shape ({{combined_features.shape}}). Expected ({{self.stimulation_samples}}, {{expected_total_channels}}). Returning zero-filled data of expected shape.")
                 return np.zeros((self.stimulation_samples, expected_total_channels))


            # Final normalization per channel (Z-score)
            for ch in range(combined_features.shape[1]):
                ch_data = combined_features[:, ch]
                ch_mean = np.mean(ch_data)
                ch_std = np.std(ch_data)

                # Prevent division by zero
                if ch_std > 1e-8:
                    combined_features[:, ch] = (ch_data - ch_mean) / ch_std
                else:
                    combined_features[:, ch] = 0.0 # Set to zero if standard deviation is zero

            # Final validation and cleaning for any NaNs or Infs introduced
            combined_features = np.where(np.isnan(combined_features), 0.0, combined_features)
            combined_features = np.where(np.isinf(combined_features), 0.0, combined_features)

            # Clip extreme values again after normalization
            combined_features = np.clip(combined_features, -5, 5)

            return combined_features

        except Exception as e:
            print(f"Error in preprocessing trial: {{e}}. Returning zero-filled data as fallback.")
            # Fallback to return zero-filled data of expected shape on general error
            expected_samples = int(config.MI_STIMULATION_DURATION * config.SAMPLING_RATE)
            expected_channels = len(self.eeg_channels) + len(self.motor_channels) # Expect 10 channels
            return np.zeros((expected_samples, expected_channels))


def preprocess_all_data_robust(X_train, X_val, X_test, cache_file=None):
    """
    Applies the robust preprocessing pipeline to training, validation, and test datasets.
    Includes caching to avoid repeated processing.
    """
    if cache_file is None:
        cache_file = config.MI_PREPROCESSED_CACHE

    # Attempt to load cached data
    if os.path.exists(cache_file):
        print("Loading cached robust preprocessed data...")
        try:
            with open(cache_file, 'rb') as f:
                processed_data = pickle.load(f)
            print("✓ Cached preprocessed data loaded successfully.")
            return processed_data
        except Exception as e:
            print(f"Error loading cached preprocessed data: {{e}}. Proceeding with fresh preprocessing.")
            # Continue to load fresh data if cache loading fails


    print("Robust preprocessing with error handling...")
    preprocessor = RobustMIPreprocessor()

    # Process training data
    X_train_processed = []
    # Corrected: Handle potential empty input data
    if X_train is not None and X_train.shape[0] > 0:
        for i in tqdm(range(len(X_train)), desc="Processing training data"):
            processed = preprocessor.preprocess_trial(X_train[i])
            X_train_processed.append(processed)
    else:
        print("Warning: Training data is empty or None. Skipping processing.")


    # Process validation data
    X_val_processed = []
    # Corrected: Handle potential empty input data
    if X_val is not None and X_val.shape[0] > 0:
        for i in tqdm(range(len(X_val)), desc="Processing validation data"):
            processed = preprocessor.preprocess_trial(X_val[i])
            X_val_processed.append(processed)
    else:
        print("Warning: Validation data is empty or None. Skipping processing.")


    # Process test data
    X_test_processed = []
    # Corrected: Handle potential empty input data
    if X_test is not None and X_test.shape[0] > 0:
        for i in tqdm(range(len(X_test)), desc="Processing test data"):
            processed = preprocessor.preprocess_trial(X_test[i])
            X_test_processed.append(processed)
    else:
        print("Warning: Test data is empty or None. Skipping processing.")


    # Convert to numpy arrays and validate shapes
    # Corrected: Handle cases where processed lists might be empty
    X_train_processed_np = np.array(X_train_processed) if X_train_processed else np.array([])
    X_val_processed_np = np.array(X_val_processed) if X_val_processed else np.array([])
    X_test_processed_np = np.array(X_test_processed) if X_test_processed else np.array([])


    # Final validation and cleaning for any NaNs or Infs
    X_train_processed_np = np.where(np.isnan(X_train_processed_np), 0.0, X_train_processed_np)
    X_val_processed_np = np.where(np.isnan(X_val_processed_np), 0.0, X_val_processed_np)
    X_test_processed_np = np.where(np.isnan(X_test_processed_np), 0.0, X_test_processed_np)

    X_train_processed_np = np.where(np.isinf(X_train_processed_np), 0.0, X_train_processed_np)
    X_val_processed_np = np.where(np.isinf(X_val_processed_np), 0.0, X_val_processed_np)
    X_test_processed_np = np.where(np.isinf(X_test_processed_np), 0.0, X_test_processed_np)


    processed_data = {
        'X_train': X_train_processed_np,
        'X_val': X_val_processed_np,
        'X_test': X_test_processed_np
    }

    # Cache the processed data
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"✓ Processed data cached successfully to {{cache_file}}")
    except Exception as e:
        print(f"Warning: Could not save processed data cache to {{cache_file}}: {{e}}")


    return processed_data

