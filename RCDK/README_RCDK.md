# ToxMix: Cytotoxicity Prediction Model

## Project Overview

This project develops a machine learning model to predict the cytotoxicity of both individual chemical compounds and their mixtures. The model uses molecular descriptors generated from the R Chemistry Development Kit (RCDK) to represent chemical structures and properties, combined with concentration data to predict cytotoxicity levels.

## Purpose

The primary purpose of this code is to:
1. Train a machine learning model that can accurately predict cytotoxicity of both single compounds and chemical mixtures
2. Evaluate the model's performance using various metrics
3. Visualize the relationship between predicted and actual cytotoxicity values
4. Create a reusable model for future cytotoxicity predictions

## Mathematical Formulas and Algorithms

### Data Preprocessing
- **Normalization**: Cytotoxicity values are clipped at 100 and then normalized to the range [0,1] using MinMaxScaler:
  ```
  normalized_value = (value - min_value) / (max_value - min_value)
  ```
- **Concentration Scaling**: Concentration values are divided by 1,000,000 to convert to a more manageable scale:
  ```
  scaled_concentration = concentration / 1000000
  ```

### Machine Learning Algorithm
- **Random Forest Regression**: An ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

### Evaluation Metrics
- **R² Score (Coefficient of Determination)**:
  ```
  R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
  ```
  Measures the proportion of variance in the dependent variable that is predictable from the independent variables.

- **Mean Absolute Error (MAE)**:
  ```
  MAE = (1/n) * Σ|y_true - y_pred|
  ```
  Measures the average magnitude of errors in a set of predictions, without considering their direction.

- **Mean Squared Error (MSE)**:
  ```
  MSE = (1/n) * Σ(y_true - y_pred)²
  ```
  Measures the average of the squares of the errors.

## Key Variables and Their Purpose

### Data Variables
- `dataset`: Main dataset containing compound names, concentrations, and cytotoxicity values
- `train`: Dataset containing RCDK descriptors for individual compounds
- `mixture_info`: JSON data containing information about chemical mixtures
- `mix_emb_df`: Calculated descriptors for mixtures using the formula_descriptors module
- `training_set`: Dataset for single compounds with their descriptors
- `testing_set`: Dataset for mixtures with their descriptors
- `altogether`: Combined dataset of both single compounds and mixtures

### Model Training Variables
- `X_train`, `y_train`: Feature matrix and target vector for training
- `X_test`, `y_test`: Feature matrix and target vector for testing
- `param_grid`: Dictionary of hyperparameters for grid search optimization
- `best_score`: Best R² score achieved during grid search
- `best_params`: Best hyperparameter combination found during grid search
- `model`: The trained Random Forest Regressor model

### Visualization Variables
- `compound_name`: Name of the compound being visualized
- `df`: Filtered dataset for a specific compound
- `concentration`: Concentration values for the selected compound
- `predicted_Cytotoxicity`: Model predictions for cytotoxicity

## Workflow Description

1. **Data Loading and Preprocessing**:
   - Load cytotoxicity data from CSV
   - Clip cytotoxicity values at 100
   - Normalize cytotoxicity values to [0,1] range
   - Scale concentration values

2. **Feature Engineering**:
   - Load RCDK descriptors for single compounds
   - Calculate descriptors for mixtures using formula_descriptors module
   - Merge descriptors with cytotoxicity data

3. **Model Training**:
   - Split data into 80% training and 20% testing sets
   - Define hyperparameter grid for Random Forest
   - Perform grid search to find optimal hyperparameters
   - Train final model with best parameters

4. **Model Evaluation**:
   - Calculate R², MAE, and MSE on test set
   - Validate model on training set
   - Save trained model to pickle file

5. **Visualization**:
   - Generate predictions for all compounds
   - Create plots comparing actual vs. predicted cytotoxicity at different concentrations
   - Visualize results for individual compounds and mixtures

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms and evaluation metrics
- matplotlib: Data visualization
- pickle: Model serialization
- formula_descriptors: Custom module for calculating descriptors for mixtures

## Files

- `train_ml_models_gridsearch.ipynb`: Main Jupyter notebook containing the code
- `data/kim_model_data_for_ml.csv`: Cytotoxicity data
- `data/KIM_rcdk_descriptors.csv`: RCDK descriptors for single compounds
- `data/mixture_data_for_mapping.json`: Information about chemical mixtures
- `formula_descriptors.py`: Module for calculating descriptors for mixtures
- `rf_80_20_point_split_rcdk.pkl`: Saved trained model

## Model Performance

The Random Forest model was optimized using grid search with the following best parameters:
- criterion: 'absolute_error'
- max_depth: 20
- n_estimators: 300
- random_state: 42

The model achieves good performance in predicting cytotoxicity for both single compounds and mixtures, as demonstrated by the visualization plots.