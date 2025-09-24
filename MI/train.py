
import os
import numpy as np
import tensorflow as tf
import pickle # Needed for loading model storage

# Import modules from the project structure
from src.config import config
from src.data_loading import load_all_mi_data
from src.preprocessing import preprocess_all_data_robust
from src.augmentation import RobustAugmenter
from models.models import FEEGNet, FEEGNet_Attention, FEEGNet_Attention_Improved # Import renamed models
from utils.training_utils import ModelStorage, train_model, evaluate_model, plot_training_history

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def main():
    print("Starting Motor Imagery EEG Classification Training")
    print("="*50)

    # 1. Load Data
    print("\nLoading Data...")
    data = load_all_mi_data(base_path=config.BASE_PATH, cache_file=config.MI_DATA_CACHE)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, test_ids = data['X_test'], data['test_ids']

    if X_train.shape[0] == 0:
         print("Error: No training data loaded. Exiting.")
         return
    if X_val.shape[0] == 0:
         print("Error: No validation data loaded. Exiting.")
         return
    if X_test.shape[0] == 0:
         print("Error: No test data loaded. Exiting.")
         return


    print(f"\n✓ Data loaded successfully")
    print(f"Training data shape: {{X_train.shape}}")
    print(f"Validation data shape: {{X_val.shape}}")
    print(f"Test data shape: {{X_test.shape}}")
    print(f"Class distribution in training: {{np.bincount(y_train)}}")
    print(f"Class distribution in validation: {{np.bincount(y_val)}}")


    # 2. Preprocess Data
    print("\nPreprocessing Data...")
    processed_data = preprocess_all_data_robust(
        X_train, X_val, X_test, cache_file=config.MI_PREPROCESSED_CACHE
    )
    X_train_proc = processed_data['X_train']
    X_val_proc = processed_data['X_val']
    X_test_proc = processed_data['X_test']

    if X_train_proc.shape[0] == 0:
         print("Error: No processed training data. Exiting.")
         return


    print(f"\n✓ Preprocessing completed")
    print(f"Processed training data shape: {{X_train_proc.shape}}")
    print(f"Processed validation data shape: {{X_val_proc.shape}}")
    print(f"Processed test data shape: {{X_test_proc.shape}}")
    # Data quality check
    print(f"Data quality check on processed data:")
    print(f"Training NaN count: {{np.sum(np.isnan(X_train_proc))}}")
    print(f"Validation NaN count: {{np.sum(np.isnan(X_val_proc))}}")
    print(f"Training inf count: {{np.sum(np.isinf(X_train_proc))}}")
    print(f"Validation inf count: {{np.sum(np.isinf(X_val_proc))}}")
    print(f"Training mean: {{np.mean(X_train_proc):.4f}}, std: {{np.std(X_train_proc):.4f}}")
    print(f"Validation mean: {{np.mean(X_val_proc):.4f}}, std: {{np.std(X_val_proc):.4f}}")


    # 3. Apply Data Augmentation
    print("\nApplying Data Augmentation...")
    augmenter = RobustAugmenter()
    X_train_aug, y_train_aug = augmenter.robust_augment(X_train_proc, y_train, n_augment=config.AUGMENTATION_FACTOR)

    if X_train_aug.shape[0] == 0:
         print("Error: No augmented training data. Exiting.")
         return

    print(f"\n✓ Data augmentation completed")
    print(f"Augmented training data shape: {{X_train_aug.shape}}")
    print(f"Augmented training labels shape: {{y_train_aug.shape}}")
    print(f"Augmented class distribution: {{np.bincount(y_train_aug)}}")


    # Reshape data for models (assuming 4D input: samples, time, channels, 1)
    print("\nReshaping data for 4D model input...")
    X_train_aug_eeg = np.expand_dims(X_train_aug, axis=-1)
    X_val_eeg = np.expand_dims(X_val_proc, axis=-1)
    X_test_eeg = np.expand_dims(X_test_proc, axis=-1) # Reshape test data for prediction

    print(f"Augmented data reshaped to: {{X_train_aug_eeg.shape}}")
    print(f"Validation data reshaped to: {{X_val_eeg.shape}}")
    print(f"Test data reshaped to: {{X_test_eeg.shape}}")


    # Get model parameters from data shape
    if X_train_aug_eeg.shape[0] > 0:
         samples, chans = X_train_aug_eeg.shape[1], X_train_aug_eeg.shape[2]
         nb_classes = len(np.unique(y_train_aug))
         print(f"Model input shape: Samples={{samples}}, Channels={{chans}}")
         print(f"Number of classes: {{nb_classes}}")
    else:
         print("Error: Cannot determine model input shape from empty augmented data. Exiting.")
         return


    # 4. Model Training and Evaluation
    model_storage = ModelStorage()
    # Attempt to load existing storage before training
    model_storage.load_all(config.MODEL_STORAGE_CACHE)

    # Define custom objects for loading saved models later
    custom_objects = {
        'FEEGNet': FEEGNet,
        'FEEGNet_Attention': FEEGNet_Attention,
        'FEEGNet_Attention_Improved': FEEGNet_Attention_Improved,
        'Multiply': tf.keras.layers.Multiply,
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
        'SeparableConv2D': tf.keras.layers.SeparableConv2D,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
        'AveragePooling2D': tf.keras.layers.AveragePooling2D,
        'max_norm': tf.keras.constraints.max_norm,
        'l2': tf.keras.regularizers.l2
    }


    # --- Train FEEGNet ---
    model_name_feegnet = 'FEEGNet'
    if model_name_feegnet not in model_storage.accuracies:
        feegnet_model = FEEGNet(
            nb_classes=nb_classes, Chans=chans, Samples=samples, dropoutRate=0.5,
            kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'
        )
        feegnet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), metrics=['accuracy'])

        print(f"\n{{model_name_feegnet}} model architecture:")
        feegnet_model.summary()

        history_feegnet = train_model(
            feegnet_model, X_train_aug_eeg, y_train_aug, X_val_eeg, y_val, model_name=model_name_feegnet
        )

        accuracy_feegnet, val_preds_feegnet, _, _ = evaluate_model(feegnet_model, X_val_eeg, y_val, model_name=model_name_feegnet)
        if history_feegnet: plot_training_history(history_feegnet, model_name_feegnet)

        # Generate and store test predictions
        if accuracy_feegnet is not None: # Only make test predictions if validation was successful
             test_preds_prob_feegnet = feegnet_model.predict(X_test_eeg)
             test_preds_classes_feegnet = np.argmax(test_preds_prob_feegnet, axis=1)
             model_storage.add_test_predictions(model_name_feegnet, test_preds_classes_feegnet)
             model_storage.add_model_results(model_name_feegnet, history_feegnet, accuracy_feegnet, val_preds_feegnet, test_preds_classes_feegnet) # Store all results
             print(f"\n✓ {{model_name_feegnet}} training completed. Accuracy: {{accuracy_feegnet:.4f}}")
        else:
             print(f"\n✗ {{model_name_feegnet}} training failed or evaluation skipped.")


    else:
        print(f"✓ Skipping {model_name_feegnet} training. Already found in storage.")
        # Load the best weights if model is in storage
        try:
            # Need to define the model architecture first before loading weights
            feegnet_model = FEEGNet(
                nb_classes=nb_classes, Chans=chans, Samples=samples, dropoutRate=0.5,
                kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'
            )
            # Load weights from the best checkpoint file
            checkpoint_path = f'{model_name_feegnet}_best.keras'
            if os.path.exists(checkpoint_path):
                 feegnet_model.load_weights(checkpoint_path)
                 print(f"✓ Loaded best weights for {model_name_feegnet} from {checkpoint_path}")
                 # Evaluate again to ensure accuracy matches storage (optional, but good practice)
                 accuracy_feegnet, val_preds_feegnet, _, _ = evaluate_model(feegnet_model, X_val_eeg, y_val, model_name=f"Loaded {model_name_feegnet}")
                 # Ensure test predictions are in storage if loading weights
                 if model_name_feegnet not in model_storage.test_predictions:
                     print(f"Generating test predictions for loaded {model_name_feegnet}...")
                     test_preds_prob_feegnet = feegnet_model.predict(X_test_eeg)
                     test_preds_classes_feegnet = np.argmax(test_preds_prob_feegnet, axis=1)
                     model_storage.add_test_predictions(model_name_feegnet, test_preds_classes_feegnet)
                     print(f"✓ Generated and stored test predictions for loaded {model_name_feegnet}")
            else:
                 print(f"Warning: Best weights checkpoint not found for {model_name_feegnet} at {checkpoint_path}. Cannot load weights.")

        except Exception as e:
            print(f"Warning: Could not load best weights for {model_name_feegnet}: {{e}}. Proceeding without loaded weights.")



    # --- Train FEEGNet+Attention ---
    model_name_attn = 'FEEGNet_Attention'
    if model_name_attn not in model_storage.accuracies:
        attn_model = FEEGNet_Attention(nb_classes, chans, samples)
        attn_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])

        print(f"\n{{model_name_attn}} model architecture:")
        attn_model.summary()

        attn_history = train_model(
            attn_model, X_train_aug_eeg, y_train_aug, X_val_eeg, y_val, model_name_attn, epochs=150 # Reduced epochs as in notebook
        )

        accuracy_attn, val_preds_attn, _, _ = evaluate_model(attn_model, X_val_eeg, y_val, model_name=model_name_attn)
        if attn_history: plot_training_history(attn_history, model_name_attn)

        # Generate and store test predictions
        if accuracy_attn is not None: # Only make test predictions if validation was successful
             test_preds_prob_attn = attn_model.predict(X_test_eeg)
             test_preds_classes_attn = np.argmax(test_preds_prob_attn, axis=1)
             model_storage.add_test_predictions(model_name_attn, test_preds_classes_attn)
             model_storage.add_model_results(model_name_attn, attn_history, accuracy_attn, val_preds_attn, test_preds_classes_attn) # Store all results
             print(f"\n✓ {{model_name_attn}} training completed. Accuracy: {{accuracy_attn:.4f}}")
        else:
             print(f"\n✗ {{model_name_attn}} training failed or evaluation skipped.")


    else:
        print(f"✓ Skipping {model_name_attn} training. Already found in storage.")
        # Load the best weights if model is in storage
        try:
            # Need to define the model architecture first before loading weights
            attn_model = FEEGNet_Attention(nb_classes, chans, samples)
            # Load weights from the best checkpoint file
            checkpoint_path = f'{model_name_attn}_best.keras'
            if os.path.exists(checkpoint_path):
                 # Custom objects needed for loading
                 attn_model.load_weights(checkpoint_path)
                 print(f"✓ Loaded best weights for {model_name_attn} from {checkpoint_path}")
                 # Evaluate again to ensure accuracy matches storage (optional)
                 accuracy_attn, val_preds_attn, _, _ = evaluate_model(attn_model, X_val_eeg, y_val, model_name=f"Loaded {model_name_attn}")
                 # Ensure test predictions are in storage if loading weights
                 if model_name_attn not in model_storage.test_predictions:
                      print(f"Generating test predictions for loaded {model_name_attn}...")
                      test_preds_prob_attn = attn_model.predict(X_test_eeg)
                      test_preds_classes_attn = np.argmax(test_preds_prob_attn, axis=1)
                      model_storage.add_test_predictions(model_name_attn, test_preds_classes_attn)
                      print(f"✓ Generated and stored test predictions for loaded {model_name_attn}")
            else:
                 print(f"Warning: Best weights checkpoint not found for {model_name_attn} at {checkpoint_path}. Cannot load weights.")

        except Exception as e:
            print(f"Warning: Could not load best weights for {model_name_attn}: {{e}}. Proceeding without loaded weights.")


    # --- Train FEEGNet+Attention (Improved) ---
    model_name_attn_imp = 'FEEGNet_Attention_Improved'
    if model_name_attn_imp not in model_storage.accuracies:
        attn_model_imp = FEEGNet_Attention_Improved(nb_classes, chans, samples)
        attn_model_imp.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])

        print(f"\n{{model_name_attn_imp}} model architecture:")
        attn_model_imp.summary()

        attn_history_imp = train_model(
            attn_model_imp, X_train_aug_eeg, y_train_aug, X_val_eeg, y_val, model_name_attn_imp, epochs=150 # Reduced epochs as in notebook
        )

        accuracy_attn_imp, val_preds_attn_imp, _, _ = evaluate_model(attn_model_imp, X_val_eeg, y_val, model_name=model_name_attn_imp)
        if attn_history_imp: plot_training_history(attn_history_imp, model_name_attn_imp)

        # Generate and store test predictions
        if accuracy_attn_imp is not None: # Only make test predictions if validation was successful
             test_preds_prob_attn_imp = attn_model_imp.predict(X_test_eeg)
             test_preds_classes_attn_imp = np.argmax(test_preds_prob_attn_imp, axis=1)
             model_storage.add_test_predictions(model_name_attn_imp, test_preds_classes_attn_imp)
             model_storage.add_model_results(model_name_attn_imp, attn_history_imp, accuracy_attn_imp, val_preds_attn_imp, test_preds_classes_attn_imp) # Store all results
             print(f"\n✓ {{model_name_attn_imp}} training completed. Accuracy: {{accuracy_attn_imp:.4f}}")
        else:
             print(f"\n✗ {{model_name_attn_imp}} training failed or evaluation skipped.")


    else:
        print(f"✓ Skipping {model_name_attn_imp} training. Already found in storage.")
        # Load the best weights if model is in storage
        try:
             # Need to define the model architecture first before loading weights
             attn_model_imp = FEEGNet_Attention_Improved(nb_classes, chans, samples)
             # Load weights from the best checkpoint file
             checkpoint_path = f'{model_name_attn_imp}_best.keras'
             if os.path.exists(checkpoint_path):
                  # Custom objects needed for loading
                  attn_model_imp.load_weights(checkpoint_path)
                  print(f"✓ Loaded best weights for {model_name_attn_imp} from {checkpoint_path}")
                  # Evaluate again to ensure accuracy matches storage (optional)
                  accuracy_attn_imp, val_preds_attn_imp, _, _ = evaluate_model(attn_model_imp, X_val_eeg, y_val, model_name=f"Loaded {model_name_attn_imp}")
                  # Ensure test predictions are in storage if loading weights
                  if model_name_attn_imp not in model_storage.test_predictions:
                       print(f"Generating test predictions for loaded {model_name_attn_imp}...")
                       test_preds_prob_attn_imp = attn_model_imp.predict(X_test_eeg)
                       test_preds_classes_attn_imp = np.argmax(test_preds_prob_attn_imp, axis=1)
                       model_storage.add_test_predictions(model_name_attn_imp, test_preds_classes_attn_imp)
                       print(f"✓ Generated and stored test predictions for loaded {model_name_attn_imp}")
             else:
                  print(f"Warning: Best weights checkpoint not found for {model_name_attn_imp} at {checkpoint_path}. Cannot load weights.")

        except Exception as e:
            print(f"Warning: Could not load best weights for {model_name_attn_imp}: {{e}}. Proceeding without loaded weights.")


    # 5. Model Comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)

    # Sort models by accuracy
    sorted_models = sorted(model_storage.accuracies.items(),
                          key=lambda x: x[1],
                          reverse=True)

    for i, (model_name, accuracy) in enumerate(sorted_models, 1):
        print(f"{{i}}. {{model_name}}: {{accuracy:.4f}}")

    print("="*50)

    # Visualize model comparison (saved to file in training_utils)
    # This plot will be generated and saved by the plot_training_history calls


    # Get top models for ensemble
    top_models = model_storage.get_top_models(config.ENSEMBLE_TOP_N)
    print(f"\nTop {{config.ENSEMBLE_TOP_N}} models for ensemble: {{top_models}}")


    # 6. Ensemble Predictions
    print("\nCreating ensemble predictions...")
    # Need to retrieve test predictions from storage for ensembling
    ensemble_predictions_list = []
    models_in_ensemble = []
    for model_name in top_models:
         test_preds = model_storage.get_test_predictions(model_name)
         if test_preds is not None:
              ensemble_predictions_list.append(test_preds)
              models_in_ensemble.append(model_name)
              print(f"  - Including test predictions from: {{model_name}}")
         else:
              print(f"  - Warning: Test predictions for {{model_name}} not found in storage. Skipping for ensemble.")

    if not ensemble_predictions_list:
         print("Error: No test predictions available from top models for ensembling. Skipping ensemble creation.")
         ensemble_submission = None
         ensemble_predictions_array = None
    else:
        ensemble_predictions_array = np.array(ensemble_predictions_list)

        # Perform majority voting
        # Corrected: Use axis=0 for voting across models for each sample
        final_predictions, counts = mode(ensemble_predictions_array, axis=0)
        final_predictions = final_predictions.flatten()

        # Convert to labels
        final_labels = ['Left' if pred == 0 else 'Right' for pred in final_predictions]

        # Create submission dataframe
        ensemble_submission = pd.DataFrame({
            'id': test_ids,
            'label': final_labels
        })

        # Print distribution
        print(f"\nEnsemble predictions distribution:")
        print(f"Left: {{np.sum(final_predictions == 0)}}")
        print(f"Right: {{np.sum(final_predictions == 1)}}")


    # 7. Save Ensemble Submission
    if ensemble_submission is not None:
        submission_filename = 'ensemble_submission.csv'
        ensemble_submission.to_csv(submission_filename, index=False)
        print(f"\n✓ Ensemble submission saved to '{{submission_filename}}'")
    else:
        print("\n✗ Ensemble submission not created due to missing test predictions.")


    # 8. Individual Model Submissions (Optional, but good for analysis)
    print("\nCreating individual model submissions...")
    # Need to retrieve test predictions from storage for individual submissions
    for model_name, predictions in model_storage.test_predictions.items():
        # Convert predictions to labels
        labels = ['Left' if pred == 0 else 'Right' for pred in predictions]

        # Create submission dataframe
        submission_df_individual = pd.DataFrame({
            'id': test_ids,
            'label': labels
        })

        # Save submission
        filename = f'{model_name}_submission.csv'
        submission_df_individual.to_csv(filename, index=False)

        # Print distribution
        print(f"\n{{model_name}} predictions:")
        print(f"  Left: {{np.sum(predictions == 0)}}")
        print(f"  Right: {{np.sum(predictions == 1)}}")
        print(f"  ✓ Saved to '{{filename}}'")


    # 9. Save Model Storage
    model_storage_file = config.MODEL_STORAGE_CACHE
    model_storage.save_all(model_storage_file)

    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nAll models trained and evaluated.")
    print(f"Model data saved to: {{model_storage_file}}")
    if ensemble_submission is not None:
        print(f"Ensemble submission saved to: ensemble_submission.csv")
    else:
        print(f"Ensemble submission creation skipped.")

    print(f"Individual model submissions saved.")
    if model_storage.accuracies:
        print("\nBest model:", model_storage.get_top_models(1)[0])
        print(f"Best accuracy: {{max(model_storage.accuracies.values()):.4f}}")
    else:
        print("\nNo models were trained or loaded.")


if __name__ == "__main__":
    main()
