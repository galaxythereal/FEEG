
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import mode
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam # Keep import in case models are compiled here

# Assuming config is imported correctly when training_utils.py is run as part of the package
from src.config import config # Corrected import based on project structure: utils is one level above src


def get_callbacks(patience_factor=1.0, model_name="model"):
    """
    Get training callbacks with configurable patience and ModelCheckpoint.

    Args:
        patience_factor (float): Multiplier for base patience values.
        model_name (str): Name of the model for saving the checkpoint file.

    Returns:
        list: List of Keras Callback objects.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=int(config.EARLY_STOPPING_PATIENCE * patience_factor),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=int(config.REDUCE_LR_PATIENCE * patience_factor),
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'{model_name}_best.keras', # Save in .keras format
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val,
                model_name="model", epochs=None, batch_size=None, patience_factor=1.0):
    """
    Train a Keras model with standard configuration and callbacks.

    Args:
        model (tf.keras.Model): The Keras model to train.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        model_name (str): Name of the model for logging and saving.
        epochs (int, optional): Number of training epochs. Defaults to config.EPOCHS.
        batch_size (int, optional): Batch size for training. Defaults to config.BATCH_SIZE.
        patience_factor (float): Multiplier for callback patience.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    print(f"\nTraining {{model_name}}...")

    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks(patience_factor=patience_factor, model_name=model_name),
            verbose=1
        )
        print(f"✓ Training of {{model_name}} completed.")
        return history
    except Exception as e:
        print(f"Error during training of {{model_name}}: {{e}}")
        # Return a dummy history object or re-raise the exception
        return None # Indicate training failed


def evaluate_model(model, X_val, y_val, model_name="model"):
    """
    Evaluate a trained model on validation data and print results.

    Args:
        model (tf.keras.Model): The trained Keras model.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        model_name (str): Name of the model for printing results.

    Returns:
        tuple: (validation_accuracy, validation_predictions, validation_predicted_classes, confusion_matrix).
               Returns (None, None, None, None) if evaluation fails or data is empty.
    """
    print(f"\nEvaluating {{model_name}}...")

    # Corrected: Handle empty validation data
    if X_val is None or X_val.shape[0] == 0:
         print(f"Warning: Validation data is empty or None for {{model_name}}. Skipping evaluation.")
         return None, None, None, None

    try:
        val_predictions = model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_accuracy = accuracy_score(y_val, val_pred_classes)

        print(f"\n{{model_name}} Validation Accuracy: {{val_accuracy:.4f}}")
        print(f"\n{{model_name}} Classification Report:")
        # Dynamically set target names based on unique classes in y_val
        unique_classes = np.unique(y_val)
        if len(unique_classes) == 2:
            target_names = ['Left', 'Right'] # Assuming binary classification
        else:
            target_names = [str(i) for i in unique_classes] # Use string representation for other cases


        print(classification_report(y_val, val_pred_classes, target_names=target_names))

        print(f"\n{{model_name}} Confusion Matrix:")
        cm = confusion_matrix(y_val, val_pred_classes)
        print(cm)

        print(f"✓ Evaluation of {{model_name}} completed.")
        return val_accuracy, val_predictions, val_pred_classes, cm

    except Exception as e:
        print(f"Error during evaluation of {{model_name}}: {{e}}")
        return None, None, None, None # Indicate evaluation failed


def plot_training_history(history, model_name="Model"):
    """
    Plot training and validation accuracy and loss history.

    Args:
        history (tf.keras.callbacks.History): Training history object.
        model_name (str): Name of the model for plot titles.
    """
    # Corrected: Handle None history object
    if history is None or not history.history:
         print(f"Warning: No history data available for plotting {{model_name}}.")
         return


    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    # Corrected: Check if keys exist in history.history
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{{model_name}} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    else:
        print(f"Warning: Accuracy data not found in history for {{model_name}}. Skipping accuracy plot.")


    # Plot Loss
    plt.subplot(1, 2, 2)
    # Corrected: Check if keys exist in history.history
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{{model_name}} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    else:
        print(f"Warning: Loss data not found in history for {{model_name}}. Skipping loss plot.")


    plt.tight_layout()
    # In a script, save the figure instead of showing it
    # Ensure models directory exists before saving
    save_dir = 'models' # Assuming models are saved in a 'models' directory relative to the main script
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{{model_name}}_training_history.png'))
    plt.close() # Close the figure to free memory
    print(f"✓ Training history plot saved for {{model_name}}.")


class ModelStorage:
    """
    Class to store and manage trained models, their histories, accuracies,
    and predictions. Supports saving and loading storage state.
    """
    def __init__(self):
        self.models = {} # Store model objects (optional, primarily store checkpoints)
        self.histories = {}
        self.accuracies = {}
        self.predictions = {} # Validation predictions (probabilities or classes)
        self.test_predictions = {} # Test predictions (classes)
   
    def add_test_predictions(self, name, predictions):
        """Add test predictions for a model"""
        self.test_predictions[name] = predictions

    def add_model_results(self, name, history, accuracy, val_predictions, test_predictions):
        """
        Add model training results and predictions to storage.

        Args:
            name (str): Name of the model.
            history (tf.keras.callbacks.History): Training history.
            accuracy (float): Validation accuracy.
            val_predictions (np.ndarray): Predictions on the validation set.
            test_predictions (np.ndarray): Predicted classes on the test set.
        """
        # self.models[name] = model # Avoid storing model objects directly
        self.histories[name] = history.history if history else None # Store history dict
        self.accuracies[name] = accuracy
        self.predictions[name] = val_predictions
        self.test_predictions[name] = test_predictions
        print(f"✓ Added results for model: {{name}}")

    def get_top_models(self, n=5):
        """
        Get the names of the top n models based on validation accuracy.

        Args:
            n (int): Number of top models to retrieve.

        Returns:
            list: List of model names (strings) sorted by accuracy descending.
        """
        if not self.accuracies:
            print("Warning: No model accuracies available for ranking.")
            return []

        # Sort models by accuracy
        sorted_models = sorted(self.accuracies.items(),
                              key=lambda x: x[1],
                              reverse=True)
        return [name for name, _ in sorted_models[:n]]

    def save_all(self, filepath):
        """
        Save model data (accuracies, predictions, history dicts) to a pickle file.
        Does NOT save the Keras model objects themselves.

        Args:
            filepath (str): Path to save the pickle file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data_to_save = {
            'histories': self.histories,
            'accuracies': self.accuracies,
            'predictions': self.predictions, # Validation predictions
            'test_predictions': self.test_predictions # Test predictions
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"✓ Model data (accuracies, predictions, histories) saved to {{filepath}}")
        except Exception as e:
            print(f"Error saving model data to {{filepath}}: {{e}}")


    def load_all(self, filepath):
        """
        Load model data (accuracies, predictions, history dicts) from a pickle file.

        Args:
            filepath (str): Path to load the pickle file from.

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.histories = data.get('histories', {})
                self.accuracies = data.get('accuracies', {})
                self.predictions = data.get('predictions', {})
                self.test_predictions = data.get('test_predictions', {})
                print(f"✓ Model data loaded from {{filepath}}")
                return True
            except Exception as e:
                print(f"Error loading model data from {{filepath}}: {{e}}. Starting with empty storage.")
                self.__init__() # Reset storage if loading fails
                return False
        else:
            print(f"Model data file not found at {{filepath}}. Starting with empty storage.")
            return False

    def get_test_predictions(self, model_name):
        """
        Get test predictions for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            np.ndarray or None: Test predictions (classes) or None if not found.
        """
        return self.test_predictions.get(model_name)

