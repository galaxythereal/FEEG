# Motor Imagery EEG Classification

This repository contains the code for a project that classifies motor imagery tasks (left vs. right hand movement) from EEG data. It utilizes deep learning models, specifically variants of EEGNet and FEEGNet, to achieve high accuracy in classification. The project is structured for reproducibility, with separate scripts for training and inference, as well as clear configuration and utility modules.



## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Running Inference](#running-inference)
- [Model Architecture](#model-architecture)
- [Expected Outputs](#expected-outputs)

## Project Structure
The repository is organized as follows:

```
.
├── models/
│   └── models.py           # Defines the FEEGNet, FEEGNet_Attention, and FEEGNet_Attention_Improved architectures
├── src/
│   ├── config.py           # Central configuration for all parameters
│   ├── data_loading.py     # Functions for loading and caching data
│   ├── preprocessing.py    # Preprocessing pipeline for EEG data
│   ├── augmentation.py     # Data augmentation techniques for EEG signals
│   └── training_utils.py   # Helper functions for training, evaluation, and model storage
├── utils/
│   ├── training_utils.py  
├── train.py                # Main script to run the training pipeline
├── inference.py      		# Script to run inference on test data
├── FEEGNet_best.keras 		# Model checkpoint
└── requirements.txt        # Python dependencies
```

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8+
- pip (Python package installer)

## Setup and Installation

Follow these steps to set up the project environment.

1. **Install Dependencies**

   Install the required Python packages using the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```
   ***Screenshot of Terminal after running pip install:***

   ![image-20250724165432734](/home/galaxy/.config/Typora/typora-user-images/image-20250724165432734.png)


3.  **Set Environment Variable for Data Path**

    The scripts require an environment variable `MTCAIC3_DATA_PATH` to be set to the root directory of the dataset.

    **On macOS/Linux:**
    
    ```bash
    export MTCAIC3_DATA_PATH=/path/to/your/mtcaic3-phase-ii
    ```
    **On Windows (Command Prompt):**
    ```bash
    set MTCAIC3_DATA_PATH=C:\path\to\your\mtcaic3-phase-ii
    ```
    **On Windows (PowerShell):**
    ```bash
    $env:MTCAIC3_DATA_PATH="C:\path\to\your\mtcaic3-phase-ii"
    ```
    To make this permanent, add it to your shell's profile file (e.g., `.bashrc`, `.zshrc`) or your system's environment variables.
    
    ***Screenshot of setting the environment variable (macOS/Linux):***
    
    ![image-20250724165533656](/home/galaxy/.config/Typora/typora-user-images/image-20250724165533656.png)

## Usage

This project has two main workflows: training the models from scratch and running inference with a pre-trained model.

### Training the Models

The `train.py` script handles the entire training pipeline, including data loading, preprocessing, augmentation, model training, and evaluation.

1.  **Run the Training Script**

    Execute the following command in your terminal:

    ```bash
    python train.py
    ```

2.  **Training Process**

    The script will perform the following steps:
    *   Load and cache the raw MI data.
    *   Preprocess and cache the data.
    *   Apply data augmentation.
    *   Train three different models: `FEEGNet`, `FEEGNet_Attention`, and `FEEGNet_Attention_Improved`.
    *   Save the best weights for each model (`.keras` files).
    *   Generate and save training history plots.
    *   Create an ensemble prediction from the top-performing models.
    *   Save individual and ensemble submission files.

    ***Screenshot of the training process starting:***
    

    ![image-20250724165601187](/home/galaxy/.config/Typora/typora-user-images/image-20250724165601187.png)
    

### Running Inference

The `inference.py` script is designed to be a standalone module for generating predictions on the test set using a pre-trained model.

1.  **Ensure Model File is Present**

    Make sure the `EEGNet_Attention_best.keras` file (or the model you wish to use) is in the root directory of the project. This file is generated during the training process.

2.  **Run the Inference Script**

    Execute the inference script from your terminal:

    ```bash
    python inference.py
    ```

3.  **Inference Process**

    The script will:
    *   Load the test data.
    *   Apply the same preprocessing steps used during training.
    *   Load the specified pre-trained model (`EEGNet_Attention_best.keras`).
    *   Generate predictions on the preprocessed test data.
    *   Save the predictions to a `inference_submission.csv` file.

    ***Screenshot of the inference process:***
    
    
    
    ![image-20250724165653171](/home/galaxy/.config/Typora/typora-user-images/image-20250724165653171.png)

## Model Architecture

The project explores three deep learning architectures based on the FEEGNet (formerly EEGNet) paper.

1.  **FEEGNet**: A compact convolutional neural network designed for EEG-based brain-computer interfaces. It includes temporal, depthwise, and separable convolutions to effectively capture spatial and temporal features from EEG signals.

2.  **FEEGNet with Attention**: This model enhances FEEGNet by incorporating a Squeeze-and-Excitation (SE) attention block. The SE block adaptively recalibrates channel-wise feature responses, allowing the model to focus on more informative channels.

3.  **Improved FEEGNet with Attention**: This version builds upon the attention model with several improvements:
    *   Increased filter sizes (`F1=16`, `F2=32`) to capture more complex features.
    *   Higher dropout rate (`0.6`) for better regularization.
    *   L2 regularization on convolutional and dense layers to prevent overfitting.
    *   An additional separable convolutional block to increase model depth.

## Expected Outputs

After running the scripts, the following files will be generated in the project's root directory:

*   **During Training (`train.py`):**
    *   `mi_data_cache.pkl`: Cached raw data.
    *   `mi_robust_preprocessed.pkl`: Cached preprocessed data.
    *   `model_storage.pkl`: A file containing accuracies, histories, and predictions for all trained models.
    *   `FEEGNet_best.keras`, `FEEGNet_Attention_best.keras`, `FEEGNet_Attention_Improved_best.keras`: The saved weights of the best performing models.
    *   `FEEGNet_training_history.png`, `FEEGNet_Attention_training_history.png`, `FEEGNet_Attention_Improved_training_history.png`: Plots of training/validation accuracy and loss.
    *   `ensemble_submission.csv`: The final submission file from the ensembled model.
    *   `FEEGNet_submission.csv`, `FEEGNet_Attention_submission.csv`, `FEEGNet_Attention_Improved_submission.csv`: Submission files from individual models.

*   **During Inference (`inference.py`):**
    *   `inference_submission.csv`: A submission file with predictions from the loaded model.

***Screenshot of the generated files after running `train.py`:***

![image-20250724170929028](/home/galaxy/.config/Typora/typora-user-images/image-20250724170929028.png)



Thanks