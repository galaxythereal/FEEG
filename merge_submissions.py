import pandas as pd
import os

def merge_submissions():
    """
    Merges the predictions from the MI and SSVEP models into a final submission file,
    keeping the original text labels.
    """
    # Define the paths to the submission files
    ssvep_submission_path = os.path.join('SSVEP', 'submission.csv')
    mi_submission_path = os.path.join('MI', 'inference_submission.csv')
    final_submission_path = 'final_submission.csv'

    # Check if the input files exist
    if not os.path.exists(ssvep_submission_path):
        print(f"Error: The SSVEP submission file was not found at {ssvep_submission_path}")
        print("Please ensure you have run the SSVEP inference script first.")
        return

    if not os.path.exists(mi_submission_path):
        print(f"Error: The MI submission file was not found at {mi_submission_path}")
        print("Please ensure you have run the MI inference script first.")
        return

    print("Loading submission files...")
    # Load the base submission file from the SSVEP task
    ssvep_df = pd.read_csv(ssvep_submission_path)

    # Load the submission file from the MI task
    mi_df = pd.read_csv(mi_submission_path)

    print(f"Base submission file loaded with {len(ssvep_df)} entries.")
    print(f"MI submission file loaded with {len(mi_df)} entries to merge.")

    # Set the 'id' column as the index for efficient updating
    ssvep_df.set_index('id', inplace=True)
    mi_df.set_index('id', inplace=True)

    # Update the SSVEP dataframe with the predictions from the MI dataframe.
    print("Merging predictions...")
    ssvep_df.update(mi_df)

    # Reset the index to turn the 'id' column back into a regular column
    final_df = ssvep_df.reset_index()

    # --- FIX ---
    # The line that converted labels to integers has been removed to keep the text format.
    # final_df['label'] = final_df['label'].astype(int) <-- This was removed.

    # Save the final merged dataframe to a new CSV file
    print(f"Saving final merged predictions to {final_submission_path}...")
    final_df.to_csv(final_submission_path, index=False)

    print("\nMerge complete!")
    print(f"The final submission file with text labels has been saved to '{final_submission_path}'.")

if __name__ == "__main__":
    merge_submissions()
