# 8020_positive-only: Cytotoxicity Prediction Model

## Overview

This repository contains a machine learning model for predicting the cytotoxicity of chemical compounds and mixtures using ChemBERTa embeddings. The model is trained on a dataset of individual compounds and their cytotoxicity measurements, and is then used to predict the cytotoxicity of chemical mixtures.

## Data Description

### Main Dataset Files

- `data/kim_model_data_for_ml.csv`: Contains cytotoxicity measurements for various compounds and mixtures at different concentrations. The dataset includes three columns:
  - `Concentration`: The concentration of the compound in μM (divided by 1,000,000 in preprocessing)
  - `Compound`: The name of the compound or mixture
  - `Cytotoxicity`: The measured cytotoxicity value (normalized to range [0,1] during preprocessing)

- `data/KIM_pubchem_hepg2_chemberta_embeddings.csv`: Contains ChemBERTa embeddings for individual compounds. These are 384-dimensional vectors that represent the chemical structure of each compound.

- `data/mixture_data_for_mapping.json`: Contains information about chemical mixtures, including:
  - `Compound Names`: List of compounds in the mixture
  - `Smiles`: SMILES notation for each compound
  - `Mole Fractions`: The mole fraction of each compound in the mixture
  - `Molecular Weights`: Molecular weight of each compound in g/mol

### Model Files

- `80_20_level_model.pkl`: The trained Random Forest model for cytotoxicity prediction

## Code Structure

### formula_descriptors.py

This module calculates molecular descriptors for chemical mixtures based on the embeddings of their constituent compounds. It contains the following functions:

- `get_component_embedding`: Retrieves the embedding for a given compound and scales it by the mole fraction
- `calculate_mixture_embedding`: Calculates the mixture embedding for a set of compounds and mole fractions
- `formulate_descriptors`: Formulates molecular descriptors for a given set of mixtures

### 8020_positive-only.ipynb

This Jupyter notebook contains the main code for data preprocessing, model training, and evaluation. The workflow is as follows:

1. **Data Loading and Preprocessing**:
   - Load cytotoxicity data from `kim_model_data_for_ml.csv`
   - Normalize cytotoxicity values to range [0,1] using MinMaxScaler
   - Scale concentration values by dividing by 1,000,000

2. **Feature Engineering**:
   - Load ChemBERTa embeddings for individual compounds
   - Generate mixture embeddings using the `formula_descriptors` module
   - Combine individual compound and mixture data

3. **Model Training**:
   - Split data into training and testing sets (80/20 split)
   - Perform grid search to find optimal hyperparameters for Random Forest Regressor
   - Train the model with the best parameters

4. **Model Evaluation**:
   - Calculate R², MAE, MSE, and RMSE on the test set
   - Perform 5-fold cross-validation
   - Generate visualizations of model performance

5. **Visualization**:
   - Plot predicted vs. actual cytotoxicity values
   - Create concentration-response curves for individual compounds and mixtures
   - Save figures for manuscript preparation

## Model Details

### Random Forest Regressor

The model uses a Random Forest Regressor with the following hyperparameters:
- `criterion`: 'absolute_error'
- `max_depth`: None (unlimited depth)
- `n_estimators`: 100
- `random_state`: 42

### Performance Metrics

The model achieves the following performance metrics on the test set:
- R² score: Approximately 0.9 (indicating good fit)
- Mean Absolute Error (MAE): Typically around 0.05-0.1
- Root Mean Squared Error (RMSE): Typically around 0.1-0.15

## Environment Requirements

To run this code, you need the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle
- functools

## Usage

1. Ensure all required data files are in the correct locations
2. Run the `8020_positive-only.ipynb` notebook to train and evaluate the model
3. The trained model will be saved as `80_20_level_model.pkl`
4. Use the model to predict cytotoxicity for new compounds or mixtures

## Visualization Examples

The notebook generates several types of visualizations:
- Concentration-response curves for individual compounds and mixtures
- Scatter plots of predicted vs. actual cytotoxicity values
- Bar plots of R² scores across cross-validation folds

## Notes

- The model uses an 80/20 train/test split at the sample level
- Cytotoxicity values are clipped at 100 before normalization
- The model is specifically designed for predicting cytotoxicity in HepG2 cells
- ChemBERTa embeddings are used to represent the chemical structure of compounds
- For mixtures, embeddings are calculated as weighted averages of individual compound embeddings based on mole fractions