# MTCAIC3_FEEG - Combined EEG Classification Project

Welcome to the main repository for the MTCAIC3 FEEG classification project. This project integrates two advanced EEG classification systems: one for Steady-State Visually Evoked Potential (SSVEP) tasks and another for Motor Imagery (MI) tasks. This central README provides a high-level guide to navigate the sub-projects and generate a final, combined submission file.

## Project Structure

The repository is organized into two main components, each addressing a specific EEG classification task, along with a script to merge their outputs.

```
MTCAIC3_FEEG/
├── SSVEP/                  # Code and models for the SSVEP Task
├── MI/                     # Code and models for the Motor Imagery Task
├── merge_submissions.py    # Script to combine the predictions from both tasks
├── Documentation/          # Contains the system description paper
└── README.md               # This main guide
```

## Complete Workflow to Generate Final Submission

Follow these steps in order to produce the final predictions. The process involves running the inference script for each sub-project and then merging the results.

### Step 1: Generate Predictions for the SSVEP Task

The SSVEP model handles the classification of trials with IDs from 4901 to 5100.

1.  **Navigate to the SSVEP directory:**
    ```bash
    cd SSVEP
    ```
2.  **Follow the instructions:** The `README.md` file in this directory contains detailed instructions for setting up the environment, installing dependencies, training the model, and running inference. The provided `SSVEP/README.md` file has all the necessary steps.
3.  **Run inference:** Execute the `inference.py` script as instructed in its README.
    ```bash
    python inference.py
    ```
4.  **Confirm the output:** This process will generate a `submission.csv` file inside the `SSVEP/` directory, containing predictions for its designated test trials.

### Step 2: Generate Predictions for the Motor Imagery (MI) Task

The MI model is responsible for classifying trials with IDs from 4901 to 5000.

1.  **Navigate to the MI directory:** From the root directory, run:
    ```bash
    cd MI
    ```
2.  **Follow the instructions:** This directory also contains a `README.md` file. Follow its guide to set up the environment and run inference.
3.  **Run inference:** Execute the `inference.py` script.
    ```bash
    python inference.py
    ```
4.  **Confirm the output:** This will create an `inference_submission.csv` file within the `MI/` directory, containing predictions for its specific set of trials.

### Step 3: Merge Submissions for the Final Result

After generating predictions from both models, you need to combine them into a single, final submission file. The `merge_submissions.py` script is designed for this purpose. It takes the predictions from the MI model (`MI/inference_submission.csv`) and updates the corresponding entries in the submission file from the SSVEP model (`SSVEP/submission.csv`).

1.  **Return to the root directory:**
    ```bash
    cd ..
    ```
2.  **Run the merge script:**
    ```bash
    python merge_submissions.py
    ```
3.  **Final Output:** The script will create a `final_submission.csv` file in the root directory. This file represents the complete and final set of predictions, ready for submission.

## Documentation

For a detailed technical description of the methodologies, models, and features used in this project, please refer to the paper located in the `Documentation/` directory:

-   `Documentation/System_description_paper.pdf`
