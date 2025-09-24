"""
Preprocessing pipeline for EEG data
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FastICA
from src.config import *

def apply_baseline_correction(stim_data, baseline_data):
    """Apply baseline correction using marker period data."""
    eeg_channels = [ch for ch in ALL_EEG_CHANNELS if ch in stim_data.columns]
    corrected_data = stim_data.copy()
    
    # Calculate baseline mean for each channel
    for ch in eeg_channels:
        baseline_mean = baseline_data[ch].mean()
        corrected_data[ch] = stim_data[ch] - baseline_mean
    
    return corrected_data

def advanced_bandpass_filter(data, low, high, fs, order=6, method='butter'):
    """Advanced bandpass filtering with multiple methods."""
    nyq = 0.5 * fs
    if method == 'butter':
        b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    elif method == 'ellip':
        b, a = signal.ellip(order, 0.1, 40, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def apply_car(data):
    """Common Average Reference."""
    return data - np.mean(data, axis=1, keepdims=True)

def apply_laplacian(data, channels=ALL_EEG_CHANNELS):
    """Spatial Laplacian filter for enhanced spatial resolution."""
    laplacian_data = data.copy()
    central_channels = ['CZ', 'PZ', 'OZ']
    
    for i, ch in enumerate(channels):
        if ch in central_channels:
            neighbors = []
            if ch == 'CZ':
                neighbors = ['FZ', 'C3', 'C4', 'PZ']
            elif ch == 'PZ':
                neighbors = ['CZ', 'PO7', 'PO8', 'OZ']
            elif ch == 'OZ':
                neighbors = ['PZ', 'PO7', 'PO8']
            
            if neighbors:
                neighbor_indices = [channels.index(n) for n in neighbors if n in channels]
                if neighbor_indices:
                    laplacian_data[:, i] = data[:, i] - np.mean(data[:, neighbor_indices], axis=1)
    
    return laplacian_data

def enhanced_preprocessing(stim_data, baseline_data=None, use_ica=True, use_laplacian=False):
    """Enhanced preprocessing pipeline with baseline correction."""
    # Apply baseline correction if baseline data provided
    if baseline_data is not None:
        data_corrected = apply_baseline_correction(stim_data, baseline_data)
    else:
        data_corrected = stim_data
    
    eeg_data = data_corrected[ALL_EEG_CHANNELS].values
    
    # 1. Robust scaling
    scaler = RobustScaler()
    eeg_scaled = scaler.fit_transform(eeg_data)
    
    # 2. Notch filter for powerline noise (50Hz + harmonics)
    for freq in [50, 100]:
        if freq < SAMPLING_RATE / 2:
            b_notch, a_notch = signal.iirnotch(freq, 30, SAMPLING_RATE)
            eeg_scaled = signal.filtfilt(b_notch, a_notch, eeg_scaled, axis=0)
    
    # 3. Bandpass filter
    eeg_filtered = advanced_bandpass_filter(eeg_scaled, 5, 45, SAMPLING_RATE, method='ellip')
    
    # 4. Spatial filtering
    if use_laplacian:
        eeg_spatial = apply_laplacian(eeg_filtered, ALL_EEG_CHANNELS)
    else:
        eeg_spatial = apply_car(eeg_filtered.T).T
    
    # 5. ICA for artifact removal
    if use_ica:
        try:
            ica = FastICA(n_components=len(ALL_EEG_CHANNELS), random_state=42, 
                         max_iter=1000, tol=1e-4)
            eeg_clean = ica.fit_transform(eeg_spatial)
        except:
            eeg_clean = eeg_spatial
    else:
        eeg_clean = eeg_spatial
    
    return pd.DataFrame(eeg_clean, columns=ALL_EEG_CHANNELS)
