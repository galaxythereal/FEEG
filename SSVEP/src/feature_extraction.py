"""
Feature extraction utilities for SSVEP classification
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from src.config import *
from src.preprocessing import enhanced_preprocessing, advanced_bandpass_filter

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyriemann"])
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

def generate_enhanced_cca_references(freq, n_samples, fs, n_harmonics=5):
    """Generate enhanced CCA reference signals with multiple harmonics."""
    t = np.arange(n_samples) / fs
    references = []
    
    for h in range(1, n_harmonics + 1):
        references.extend([
            np.sin(2 * np.pi * h * freq * t),
            np.cos(2 * np.pi * h * freq * t)
        ])
    
    return np.array(references).T

def task_related_component_analysis(eeg_data, template):
    """Enhanced TRCA implementation."""
    try:
        # Compute cross-covariance between trial and template
        cross_cov = np.cov(eeg_data.flatten(), template.flatten())[0, 1]
        
        # Compute spatial correlation
        spatial_corr = []
        for ch in range(eeg_data.shape[0]):
            r, _ = pearsonr(eeg_data[ch, :], template[ch, :])
            spatial_corr.append(r if not np.isnan(r) else 0)
        
        return {
            'cross_correlation': cross_cov,
            'mean_spatial_correlation': np.mean(spatial_corr),
            'max_spatial_correlation': np.max(spatial_corr),
            'spatial_correlation_std': np.std(spatial_corr)
        }
    except:
        return {
            'cross_correlation': 0,
            'mean_spatial_correlation': 0,
            'max_spatial_correlation': 0,
            'spatial_correlation_std': 0
        }

def multiband_cca_analysis(eeg_data, fs):
    """Multi-band CCA analysis with weighted combination."""
    results = {}
    
    for i, (low, high) in enumerate(FILTER_BANKS):
        try:
            eeg_band = advanced_bandpass_filter(eeg_data.T, low, high, fs).T
            
            for label, freq in SSVEP_FREQS.items():
                ref_signals = generate_enhanced_cca_references(freq, eeg_data.shape[1], fs)
                
                try:
                    cca = CCA(n_components=1, max_iter=1000, tol=1e-6)
                    cca.fit(eeg_band.T, ref_signals)
                    x_c, y_c = cca.transform(eeg_band.T, ref_signals)
                    
                    correlation = np.corrcoef(x_c.flatten(), y_c.flatten())[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                    
                    results[f'cca_{label}_band{i+1}'] = correlation
                    results[f'cca_{label}_band{i+1}_power'] = np.var(x_c)
                    
                except:
                    results[f'cca_{label}_band{i+1}'] = 0
                    results[f'cca_{label}_band{i+1}_power'] = 0
        except:
            for label in SSVEP_FREQS.keys():
                results[f'cca_{label}_band{i+1}'] = 0
                results[f'cca_{label}_band{i+1}_power'] = 0
    
    return results

def riemannian_features(eeg_data):
    """Extract Riemannian manifold features."""
    try:
        eeg_reshaped = eeg_data.reshape(1, eeg_data.shape[0], eeg_data.shape[1])
        
        cov_estimator = Covariances(estimator='oas')
        cov_matrix = cov_estimator.transform(eeg_reshaped)[0]
        
        ts = TangentSpace(metric='riemann')
        ts.fit(cov_matrix.reshape(1, *cov_matrix.shape))
        tangent_features = ts.transform(cov_matrix.reshape(1, *cov_matrix.shape))[0]
        
        return {
            f'riemannian_feature_{i}': feat
            for i, feat in enumerate(tangent_features[:20])
        }
    except:
        return {f'riemannian_feature_{i}': 0 for i in range(20)}

def spectral_features(eeg_data, fs):
    """Extract advanced spectral features."""
    features = {}
    
    for i, ch in enumerate(ALL_EEG_CHANNELS):
        try:
            freqs, psd = signal.welch(eeg_data[i, :], fs, nperseg=fs)
            
            # Extract power in SSVEP frequency bands
            for label, freq in SSVEP_FREQS.items():
                freq_mask = (freqs >= freq - 0.5) & (freqs <= freq + 0.5)
                if np.any(freq_mask):
                    features[f'psd_{label}_{ch}'] = np.mean(psd[freq_mask])
                else:
                    features[f'psd_{label}_{ch}'] = 0
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            features[f'spectral_entropy_{ch}'] = -np.sum(psd_norm * np.log(psd_norm))
            
        except:
            for label in SSVEP_FREQS.keys():
                features[f'psd_{label}_{ch}'] = 0
            features[f'spectral_entropy_{ch}'] = 0
    
    return features

def extract_comprehensive_features(stim_data, baseline_data, templates):
    """Extract comprehensive feature set from stimulation period only."""
    features = {}
    
    # Preprocess EEG data with baseline correction
    eeg_clean = enhanced_preprocessing(stim_data, baseline_data, use_ica=True, use_laplacian=False)
    eeg_data = eeg_clean.T.values
    motion_data = stim_data[MOTION_CHANNELS]
    
    # 1. TRCA features
    for label, template in templates.items():
        trca_feats = task_related_component_analysis(eeg_data, template)
        for key, val in trca_feats.items():
            features[f'trca_{label}_{key}'] = val
    
    # 2. Multi-band CCA features
    cca_features = multiband_cca_analysis(eeg_data, SAMPLING_RATE)
    features.update(cca_features)
    
    # 3. Riemannian features
    riem_features = riemannian_features(eeg_data)
    features.update(riem_features)
    
    # 4. Spectral features
    spectral_feats = spectral_features(eeg_data, SAMPLING_RATE)
    features.update(spectral_feats)
    
    # 5. Statistical features from motion data
    for ch in MOTION_CHANNELS:
        ch_data = motion_data[ch].values
        features.update({
            f'{ch}_mean': np.mean(ch_data),
            f'{ch}_std': np.std(ch_data),
            f'{ch}_skew': pd.Series(ch_data).skew(),
            f'{ch}_kurt': pd.Series(ch_data).kurtosis(),
            f'{ch}_energy': np.sum(ch_data ** 2)
        })
    
    # 6. EEG statistical features
    for i, ch in enumerate(ALL_EEG_CHANNELS):
        ch_data = eeg_data[i, :]
        features.update({
            f'eeg_{ch}_power': np.var(ch_data),
            f'eeg_{ch}_peak2peak': np.ptp(ch_data),
            f'eeg_{ch}_rms': np.sqrt(np.mean(ch_data ** 2))
        })
    
    return features
