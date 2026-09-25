# Loan Approval Project

## Overview
This project predicts loan approval decisions using applicant financial information and machine-learning classification models.
It includes:
- A terminal-based prediction workflow
- A desktop GUI
- Batch CSV prediction
- Seven machine-learning models
- Evaluation metrics
- Confusion matrices
- Model comparison charts
- Risk categories
- Prediction explanations
- Feature importance
- Cross-validation
- Prediction history
- Dataset backups
- Application addition and removal

## Requirements
Use Python 3.10 or newer.
Install the required packages:
py -3 -m pip install pandas scikit-learn matplotlib seaborn

Tkinter is included with the standard Windows Python installation.
Confirm Python is available:
py -3 --version

Open the terminal inside the project folder:
cd "C:\Users\William\OneDrive\.vscode\loan_approval_project"

## Project Files
loan_approval_dataset.csv
The labeled training dataset.
Required fields include:
- loan_id
- no_of_dependents
- education
- self_employed
- income_annum
- loan_amount
- loan_term
- cibil_score
- residential_assets_value
- commercial_assets_value
- luxury_assets_value
- bank_asset_value
- loan_status

loan_status must contain:
Approved
Rejected

The dataset does not require an approval_probability column.

loan_approval_predict.py
Used for predicting new applicants.
This script uses the optimized Random Forest model and one decision threshold:
Probability >= 0.50: Approved
Probability < 0.50: Rejected

It also provides:
- Input validation
- Approval probability
- Risk category
- Plain-language explanation
- Prediction history
- Optional dataset updates
- Application removal
- Automatic backups

loan_approval_models.py
Runs the complete seven-model evaluation.
The seven models are:
- Optimized Random Forest
- Logistic Regression
- Decision Tree
- Support Vector Machine
- K-Nearest Neighbors
- Extra Trees
- AdaBoost

loan_approval_optimized.py
Contains the optimized training and evaluation implementation used by models.py.

loan_approval_gui.py
Opens the desktop application form.

loan_approval_utils.py
Contains shared validation, backup, risk-category, explanation, history, and feature-importance functions.

## Run the Desktop GUI
py -3 loan_approval_gui.py

Enter the applicant information and select Predict.
The GUI displays:
- Approval probability
- Approval or rejection decision
- Risk category
- Explanation of influential applicant factors
Close the window to stop the program.

## Run Interactive Terminal Prediction
py -3 loan_approval_predict.py

The terminal asks how many applications you want to enter and then requests each field.
After prediction, it asks whether to add the applications to the training dataset.
Answer:
y
to append them, or:
n
to keep them separate.
The script then asks whether you want to remove applications from the dataset.

## Predict From a Separate CSV
Create a CSV containing applicant feature columns without loan_status.
Run:
py -3 loan_approval_predict.py --input-csv new_applications.csv

Save the predictions to a separate results file:
py -3 loan_approval_predict.py ^
  --input-csv new_applications.csv ^
  --output-csv predictions.csv

The output contains:
- Applicant information
- approval_probability
- loan_decision
- risk_category
- explanation

The probability is displayed as a decimal rounded to two places, such as:
0.58
0.36

## Save Predictions to History
Prediction history is automatically stored in:
prediction_history.csv

Use another history file if needed:
py -3 loan_approval_predict.py ^
  --input-csv new_applications.csv ^
  --history-file my_prediction_history.csv

## Add Applications to the Training Dataset
To automatically append applications:
py -3 loan_approval_predict.py ^
  --input-csv new_applications.csv ^
  --save-to-training-data

A timestamped backup is created before the dataset is changed.

Example backup:
loan_approval_dataset.backup_20260923_195258.csv

## Remove Applications
Remove applications by loan_id directly:
py -3 loan_approval_predict.py --remove-ids 12 25 31

The script displays the selected rows and asks for confirmation.
The normal prediction workflow also asks whether to remove applications at the end. You can enter row numbers such as:
1 5 20
or:
1,5,20

The valid range is displayed based on the current number of dataset rows.

## Run Model Evaluation
Run the standard evaluation:
py -3 loan_approval_models.py

This produces:
- Evaluation metrics
- Confusion matrices
- Confusion-matrix charts
- Feature-importance CSV
- Metrics CSV
- Confusion-matrix text report

The default report files use the prefix:
loan_model_report

Run with a custom report prefix:
py -3 loan_approval_models.py --report-prefix september_report

## Evaluation Metrics
The evaluation includes:
- Accuracy
- Balanced accuracy
- Precision
- Recall
- Specificity
- Negative predictive value
- F1-score
- Matthews correlation coefficient
- ROC AUC
- Precision-recall AUC
- Log loss

## Cross-Validation
Cross-validation is optional because it takes additional time:
py -3 loan_approval_models.py --cross-validation

It evaluates the optimized Random Forest across three folds and reports:
- Mean accuracy
- Accuracy standard deviation
- Mean balanced accuracy
- Mean ROC AUC

## Important Notes
- Do not include approval_probability as a training feature. It is a model output, not an original applicant attribute.
- Keep loan_status in the training dataset for supervised learning.
- New unlabeled applications can be predicted, but they cannot produce meaningful accuracy metrics until their real outcomes are known.
- Keep backup files until you are certain dataset changes are correct.
- The optimized evaluation script may calculate a validation-selected threshold for benchmarking, while predict.py uses the fixed 0.50 applicant decision rule.
