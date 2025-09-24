"""
Data augmentation techniques for EEG signals
"""
import numpy as np
import pandas as pd
from src.config import ALL_EEG_CHANNELS

def augment_trial_data(stim_data, noise_factor=0.05, time_shift_samples=25):
    """Advanced data augmentation techniques for stimulation data."""
    augmented_trials = [stim_data]  # Original
    
    # 1. Gaussian noise augmentation
    noisy_trial = stim_data.copy()
    noise = np.random.normal(0, noise_factor, noisy_trial[ALL_EEG_CHANNELS].shape)
    noisy_trial[ALL_EEG_CHANNELS] += noise
    augmented_trials.append(noisy_trial)
    
    # 2. Time shifting (only if enough samples)
    if len(stim_data) > time_shift_samples * 2:
        shifted_trial = stim_data.iloc[time_shift_samples:-time_shift_samples].copy()
        shifted_trial.index = range(len(shifted_trial))
        # Pad to original length
        padding = pd.DataFrame(np.zeros((time_shift_samples * 2, len(stim_data.columns))),
                             columns=stim_data.columns)
        shifted_trial = pd.concat([shifted_trial, padding], ignore_index=True)[:len(stim_data)]
        augmented_trials.append(shifted_trial)
    
    # 3. Amplitude scaling
    scaled_trial = stim_data.copy()
    scale_factor = np.random.uniform(0.9, 1.1, len(ALL_EEG_CHANNELS))
    for i, ch in enumerate(ALL_EEG_CHANNELS):
        scaled_trial[ch] *= scale_factor[i]
    augmented_trials.append(scaled_trial)
    
    return augmented_trials
