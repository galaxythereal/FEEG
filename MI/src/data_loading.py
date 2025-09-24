
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from .config import config # Import config from the local src directory

def convert_adc_to_uv(trial_data):
    """
    Convert raw 24-bit ADC values to microvolts.
    Handles sign extension ±750 mV over 24-bit range.
    """
    data = np.asarray(trial_data)
    data24 = data.astype(np.int64, copy=False) & 0xFFFFFF

    # Perform sign-extension
    signed = np.where(data24 & 0x800000,
                      data24 - (1 << 24),
                      data24)

    # Scale to volts: ±0.75 V over 2^24
    volts = signed * (1.5 / (1 << 24))
    return volts * 1e6  # convert to µV

def load_trial_data(row, base_path=None):
    """Load EEG data for a specific trial"""
    if base_path is None:
        base_path = config.BASE_PATH

    # Determine dataset type based on ID range
    id_num = row['id']
    if id_num <= 4800:
        dataset = 'train'
    elif id_num <= 4900:
        dataset = 'validation'
    else:
        dataset = 'test'

    # Construct the path to EEGdata.csv
    eeg_path = os.path.join(base_path, row['task'], dataset, row['subject_id'], str(row['trial_session']), 'EEGdata.csv')


    # Load the entire EEG file
    try:
        eeg_data = pd.read_csv(eeg_path)
    except FileNotFoundError:
        print(f"Error: EEG data file not found at {{eeg_path}}")
        return None
    except Exception as e:
        print(f"An error occurred reading {{eeg_path}}: {{e}}")
        return None


    # Calculate indices for the specific trial
    trial_num = int(row['trial'])
    if row['task'] == 'MI':
        samples_per_trial = int(config.MI_TRIAL_DURATION * config.SAMPLING_RATE)
    else:  # SSVEP
        samples_per_trial = int(config.SSVEP_TRIAL_DURATION * config.SAMPLING_RATE)


    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial

    # Extract the trial data (only EEG channels)
    # Ensure we handle potential errors if columns are missing or index is out of bounds
    try:
        # Check if channels exist before selecting
        missing_channels = [ch for ch in config.EEG_CHANNELS if ch not in eeg_data.columns]
        if missing_channels:
             print(f"Error: Missing expected EEG channels in {{eeg_path}}: {{missing_channels}}. Available columns: {{eeg_data.columns.tolist()}}")
             return None

        trial_data = eeg_data[config.EEG_CHANNELS].iloc[start_idx:end_idx].values

        # Check if the extracted data has the expected number of samples
        if trial_data.shape[0] != samples_per_trial:
             print(f"Warning: Extracted trial data has unexpected number of samples. Expected {{samples_per_trial}}, got {{trial_data.shape[0]}} from {{eeg_path}}")
             # Pad or truncate if necessary to match expected size? Or return None?
             # For robustness, let's pad with zeros if too short
             if trial_data.shape[0] < samples_per_trial:
                 padding = samples_per_trial - trial_data.shape[0]
                 trial_data = np.pad(trial_data, ((0, padding), (0, 0)), mode='constant') # Pad with zeros
             elif trial_data.shape[0] > samples_per_trial:
                 trial_data = trial_data[:samples_per_trial, :] # Truncate

    except IndexError as e:
        print(f"Error: Index out of bounds when loading data from {{eeg_path}}. Start index: {{start_idx}}, End index: {{end_idx}}, Data length: {{len(eeg_data)}}. Error: {{e}}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while extracting trial data from {{eeg_path}}: {{e}}")
        return None


    return trial_data

def load_all_mi_data(base_path=None, cache_file=None):
    """Load all MI data with caching and progress bar"""
    if base_path is None:
        base_path = config.BASE_PATH
    if cache_file is None:
        cache_file = config.MI_DATA_CACHE

    if os.path.exists(cache_file):
        print("Loading cached data...")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print("✓ Cached data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading cached data: {{e}}. Loading fresh data instead.")
            # Continue to load fresh data if cache loading fails


    print("Loading fresh data...")
    # Load index files
    try:
        train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
        validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    except FileNotFoundError as e:
        print(f"Error: Index file not found: {{e}}. Please ensure data is in the correct location ({{base_path}}).")
        # Return empty data structures or raise an error
        return {
            'X_train': np.array([]), 'y_train': np.array([]),
            'X_val': np.array([]), 'y_val': np.array([]),
            'X_test': np.array([]), 'test_ids': np.array([]),
            'train_df': pd.DataFrame(), 'val_df': pd.DataFrame(), 'test_df': pd.DataFrame()
        }
    except Exception as e:
        print(f"An error occurred reading an index file: {{e}}")
        return {
            'X_train': np.array([]), 'y_train': np.array([]),
            'X_val': np.array([]), 'y_val': np.array([]),
            'X_test': np.array([]), 'test_ids': np.array([]),
            'train_df': pd.DataFrame(), 'val_df': pd.DataFrame(), 'test_df': pd.DataFrame()
        }


    # Filter for MI task only
    train_mi = train_df[train_df['task'] == 'MI'].copy()
    validation_mi = validation_df[validation_df['task'] == 'MI'].copy()
    test_mi = test_df[test_df['task'] == 'MI'].copy()

    # Load training data
    X_train, y_train = [], []
    # Corrected: Ensure correct total count for tqdm if rows were filtered
    for idx, row in tqdm(train_mi.iterrows(), total=len(train_mi), desc="Loading training MI data"):
        trial_data = load_trial_data(row, base_path)
        if trial_data is not None: # Only process if data loading was successful
            trial_uv = convert_adc_to_uv(trial_data)
            X_train.append(trial_uv)
            # Corrected: Ensure label exists and is handled
            if 'label' in row:
                 y_train.append(1 if row['label'] == 'Right' else 0)
            else:
                 print(f"Warning: Missing label for training trial with ID {{row['id']}}. Skipping.")


    # Load validation data
    X_val, y_val = [], []
    # Corrected: Ensure correct total count for tqdm if rows were filtered
    for idx, row in tqdm(validation_mi.iterrows(), total=len(validation_mi), desc="Loading validation MI data"):
        trial_data = load_trial_data(row, base_path)
        if trial_data is not None: # Only process if data loading was successful
            trial_uv = convert_adc_to_uv(trial_data)
            X_val.append(trial_uv)
            # Corrected: Ensure label exists and is handled
            if 'label' in row:
                 y_val.append(1 if row['label'] == 'Right' else 0)
            else:
                 print(f"Warning: Missing label for validation trial with ID {{row['id']}}. Skipping.")


    # Load test data
    X_test, test_ids_list = [], []
    # Corrected: Ensure correct total count for tqdm if rows were filtered
    for idx, row in tqdm(test_mi.iterrows(), total=len(test_mi), desc="Loading test MI data"):
        trial_data = load_trial_data(row, base_path)
        if trial_data is not None: # Only process if data loading was successful
            trial_uv = convert_adc_to_uv(trial_data)
            X_test.append(trial_uv)
            test_ids_list.append(row['id'])
        else:
            print(f"Skipping test trial with ID {{row['id']}} due to data loading error.")


    # Convert to numpy arrays
    # Handle cases where lists might be empty due to loading errors
    X_train_np = np.array(X_train) if X_train else np.array([])
    y_train_np = np.array(y_train) if y_train else np.array([])
    X_val_np = np.array(X_val) if X_val else np.array([])
    y_val_np = np.array(y_val) if y_val else np.array([])
    X_test_np = np.array(X_test) if X_test else np.array([])
    test_ids_np = np.array(test_ids_list) if test_ids_list else np.array([])


    data = {
        'X_train': X_train_np, 'y_train': y_train_np,
        'X_val': X_val_np, 'y_val': y_val_np,
        'X_test': X_test_np, 'test_ids': test_ids_np,
        'train_df': train_mi, 'val_df': validation_mi, 'test_df': test_mi # Include filtered dataframes
    }

    # Cache the data
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Data cached successfully to {{cache_file}}")
    except Exception as e:
        print(f"Warning: Could not save data cache to {{cache_file}}: {{e}}")


    return data

