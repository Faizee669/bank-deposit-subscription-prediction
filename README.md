# Bank Term Deposit Prediction

This repository contains a machine learning project for predicting whether a client will subscribe to a bank term deposit. The goal is to build a binary classification model using the **Playground Series S5E8: Bank Term Deposit Prediction** dataset.

## Project Overview

- **Objective:** Predict client subscription (`y = 1` or `0`) to a bank's term deposit.
- **Evaluation Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve)
- **Model:** LightGBM gradient boosting machine with stratified 5-fold cross-validation.
- **Features:** Demographic, campaign-related data along with engineered features for improved performance.

## Pipeline Steps

1. Load and inspect the data  
2. Exploratory Data Analysis (EDA)  
3. Handle missing values and encode categorical variables  
4. Feature engineering including interaction terms and transformed variables  
5. Train LightGBM model with hyperparameter tuning and early stopping  
6. Generate out-of-fold predictions and final test predictions  
7. Export submission file for competition

## Usage

To run the project:

1. Ensure you have Python 3 and required packages installed (`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `lightgbm`).
2. Place the dataset CSV files (`train.csv`, `test.csv`) in the appropriate directory.
3. Run the notebook or script to reproduce training, evaluation, and prediction.

## Repository Structure

- `lightgbm-with-cross-validation.ipynb`: The main analysis and modeling notebook.
- `train.csv` and `test.csv`: Input datasets (not included due to size/license).
- `submission.csv`: Sample output prediction file after model inference.

## Contacts & Links

Feel free to open issues or submit pull requests for improvements or questions.

---

This project is part of the Kaggle Playground Series 2025.

---


