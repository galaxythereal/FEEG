
import os
import numpy as np
from tqdm import tqdm
from .config import config # Corrected import for package structure

class RobustAugmenter:
    """
    Applies robust data augmentation techniques to EEG data.

    Includes adding Gaussian noise, time shifting, and amplitude scaling.
    Designed to handle potential errors and maintain data shape.
    """
    def __init__(self):
        # Assuming config is imported correctly when augmentation.py is run as part of the package
        self.noise_factor = config.NOISE_FACTOR
        self.time_shift_max = config.TIME_SHIFT_MAX
        self.amplitude_scale_range = config.AMPLITUDE_SCALE_RANGE

    def add_gaussian_noise(self, data, noise_factor=None):
        """Add Gaussian noise to the data with validation."""
        if noise_factor is None:
            noise_factor = self.noise_factor

        # Corrected: Handle empty data
        if data.shape[0] == 0:
             print("Warning: Data is empty, skipping noise augmentation.")
             return data

        try:
            # Calculate noise based on the standard deviation of the data
            data_std = np.std(data)
            # Prevent division by zero or extremely small std
            if data_std > 1e-8:
                noise = np.random.normal(0, noise_factor * data_std, data.shape)
                augmented_data = data + noise
                # Validate for NaNs/Infs introduced by noise
                if np.any(np.isnan(augmented_data)) or np.any(np.isinf(augmented_data)):
                     print("Warning: Noise augmentation resulted in NaN or Inf. Returning original data.")
                     return data
                return augmented_data
            else:
                print("Warning: Data standard deviation is zero or too small, skipping noise augmentation.")
                return data
        except Exception as e:
            print(f"Error during noise augmentation: {{e}}. Returning original data.")
            return data

    def time_shift(self, data, max_shift=None):
        """Apply time shifting to the data with validation."""
        if max_shift is None:
            max_shift = self.time_shift_max

        # Corrected: Handle empty data or zero max_shift
        if data.shape[0] == 0 or max_shift == 0:
             if data.shape[0] == 0:
                 print("Warning: Data is empty, skipping time shift augmentation.")
             return data

        try:
            # Ensure max_shift is not larger than the data length
            if max_shift >= data.shape[0]:
                 print(f"Warning: Max time shift ({{max_shift}}) is >= data length ({{data.shape[0]}}). Using data length - 1 as max shift.")
                 max_shift = data.shape[0] - 1
                 if max_shift < 0: # Should not happen if data length > 0, but as a safeguard
                     print("Warning: Data length is 1, cannot apply time shift. Returning original data.")
                     return data


            shift = np.random.randint(-max_shift, max_shift + 1)

            if shift > 0:
                augmented_data = np.pad(data, ((shift, 0), (0, 0)), mode='edge')[:-shift]
            elif shift < 0:
                augmented_data = np.pad(data, ((0, -shift), (0, 0)), mode='edge')[-shift:]
            else: # shift == 0
                augmented_data = data # No shift needed


            # Validate augmented data shape
            if augmented_data.shape != data.shape:
                 print(f"Warning: Time shift resulted in shape mismatch. Expected {{data.shape}}, got {{augmented_data.shape}}. Returning original data.")
                 return data

            # Validate for NaNs/Infs introduced by padding
            if np.any(np.isnan(augmented_data)) or np.any(np.isinf(augmented_data)):
                 print("Warning: Time shift augmentation resulted in NaN or Inf. Returning original data.")
                 return data

            return augmented_data

        except Exception as e:
            print(f"Error during time shift augmentation: {{e}}. Returning original data.")
            return data

    def amplitude_scale(self, data, scale_range=None):
        """Apply amplitude scaling to the data with validation."""
        if scale_range is None:
            scale_range = self.amplitude_scale_range

        # Corrected: Handle empty data
        if data.shape[0] == 0:
             print("Warning: Data is empty, skipping amplitude scaling augmentation.")
             return data

        try:
            # Ensure scale range is valid
            if scale_range < 0:
                 print(f"Warning: Amplitude scale range ({{scale_range}}) is negative. Using absolute value.")
                 scale_range = abs(scale_range)

            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            augmented_data = data * scale

            # Clip extreme values after scaling
            augmented_data = np.clip(augmented_data, -10, 10)

            # Validate for NaNs/Infs introduced by scaling
            if np.any(np.isnan(augmented_data)) or np.any(np.isinf(augmented_data)):
                 print("Warning: Amplitude scale augmentation resulted in NaN or Inf. Returning original data.")
                 return data

            return augmented_data
        except Exception as e:
            print(f"Error during amplitude scale augmentation: {{e}}. Returning original data.")
            return data

    def robust_augment(self, X, y, n_augment=None):
        """
        Applies robust augmentation to a dataset.

        Augments each sample multiple times with different techniques
        and ensures consistent shapes and valid data.
        """
        if n_augment is None:
            n_augment = config.AUGMENTATION_FACTOR

        X_aug = []
        y_aug = []

        # Add original data to the augmented set
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])

        # Add augmented data
        # Corrected: Handle empty input data
        if X is None or len(X) == 0:
             print("Warning: Input data for augmentation is empty or None. Returning original data only.")
             return np.array(X_aug), np.array(y_aug)


        for i in tqdm(range(len(X)), desc="Applying Robust Augmentation"):
            for _ in range(n_augment):
                aug_data = X[i].copy() # Start with a copy of the original data

                # Apply augmentations with a certain probability to add variation
                if np.random.rand() < 0.7: # Apply noise with 70% probability
                    aug_data = self.add_gaussian_noise(aug_data, config.NOISE_FACTOR)

                if np.random.rand() < 0.7: # Apply time shift with 70% probability
                    aug_data = self.time_shift(aug_data, config.TIME_SHIFT_MAX)

                if np.random.rand() < 0.7: # Apply amplitude scaling with 70% probability
                    aug_data = self.amplitude_scale(aug_data, config.AMPLITUDE_SCALE_RANGE)

                # Validate augmented data before adding to the list
                # Ensure the shape is correct and no NaNs/Infs were introduced
                if aug_data.shape == X[i].shape and not np.any(np.isnan(aug_data)) and not np.any(np.isinf(aug_data)):
                    X_aug.append(aug_data)
                    y_aug.append(y[i])
                else:
                    print(f"Warning: Augmented data validation failed for sample {i}. Shape: {{aug_data.shape}}, NaN: {{np.any(np.isnan(aug_data))}}, Inf: {{np.any(np.isinf(aug_data))}}. Skipping this augmented variant.")


        X_aug_np = np.array(X_aug)
        y_aug_np = np.array(y_aug)

        return X_aug_np, y_aug_np

