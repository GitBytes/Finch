# ToxMix: Cytotoxicity Prediction Model for Chemical Mixtures

This README explains the code, variables, and environment used in the `8020_positive-only.ipynb` notebook, which implements a machine learning model for predicting the cytotoxicity of both individual chemicals and chemical mixtures.

## Overview

The notebook implements a Random Forest Regression model that predicts cytotoxicity values based on chemical embeddings. The model is trained on individual chemical compounds and then tested on both individual compounds and chemical mixtures. The "8020" in the filename refers to the 80/20 train/test split used in the model evaluation.

## Environment Setup

### Dependencies

The notebook requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle
- formula_descriptors (custom module)

### Data Files

The notebook uses several data files:
- `data/kim_model_data_for_ml.csv`: Contains cytotoxicity data for compounds at different concentrations
- `data/KIM_pubchem_hepg2_chemberta_embeddings_PRETRAINED.csv`: Contains ChemBERTa embeddings for individual compounds
- `data/mixture_data_for_mapping.json`: Contains information about chemical mixtures, including compound names and mole fractions

## Code Structure and Workflow

### 1. Data Loading and Preprocessing

```python
# Load cytotoxicity data
dataset = pd.read_csv("data/kim_model_data_for_ml.csv")

# Normalize cytotoxicity values to [0,1] range
dataset['Cytotoxicity'] = dataset['Cytotoxicity'].clip(upper=100)
scaler = MinMaxScaler()
dataset["Cytotoxicity"] = scaler.fit_transform(dataset["Cytotoxicity"].values.reshape(-1, 1))

# Convert concentration from μM to M
dataset["Concentration"] = dataset["Concentration"]/1000000
```

### 2. Feature Engineering

The notebook uses ChemBERTa embeddings as features for the model:

```python
# Load embeddings for individual compounds
train = pd.read_csv("data/KIM_pubchem_hepg2_chemberta_embeddings_PRETRAINED.csv", index_col=0)
train = train[["compound"]+[ "emb_{}".format(i) for i in range(0,384) ]]
train.set_index('compound', inplace=True)

# Merge embeddings with cytotoxicity data
training_set = dataset.merge(train, left_on='Compound', right_index=True)
```

For chemical mixtures, the notebook uses the `formula_descriptors` module to calculate embeddings:

```python
# Load mixture information
mixture_info = pd.read_json('data/mixture_data_for_mapping.json', orient='index')
mixture_info = mixture_info.drop(index=['Mix15', 'Mix3'])

# Calculate embeddings for mixtures
mix_emb_df = formula_descriptors.formulate_descriptors(mixture_info, train)

# Merge mixture embeddings with cytotoxicity data
testing_set = dataset.merge(mix_emb_df, left_on='Compound', right_index=True)
```

### 3. Model Training and Evaluation

The notebook uses a Random Forest Regressor model with hyperparameter tuning:

```python
# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'criterion': ['squared_error', 'absolute_error'],
    'random_state': [42]
}

# Find best parameters using grid search
for params in ParameterGrid(param_grid):
    rf.set_params(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = r2_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_params = params
        model = rf
```

The final model uses:
```python
best = RandomForestRegressor(criterion='squared_error', max_depth=10, n_estimators=100, random_state=42)
```

### 4. Model Validation

The notebook performs 5-fold cross-validation to assess model performance:

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    best_model.fit(X_train_fold, y_train_fold)
    y_val_pred = best_model.predict(X_val_fold)
    # Calculate metrics
    r2 = r2_score(y_val_fold, y_val_pred)
    mae = mean_absolute_error(y_val_fold, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
```

### 5. Model Persistence

The trained model is saved to disk for later use:

```python
pkl_file = '80_20_level_model_pretrained.pkl'
with open(pkl_file, 'wb') as file:
    pickle.dump(model, file)
```

### 6. Visualization

The notebook includes several visualization functions to analyze model performance:

```python
# Plot predicted vs actual cytotoxicity for individual compounds
def plot_compound(compound_name, dataframe):
    df = dataframe[dataframe["Compound"] == compound_name]
    plt.plot([str(i*1000000) for i in df.Concentration], df['Cytotoxicity']*100, 
             color='blue', marker='o', linestyle='-', label='Cytotoxicity')
    plt.plot([str(i*1000000) for i in df.Concentration], df['predicted_Cytotoxicity']*100, 
             color='green', marker='x', linestyle='--', label='Predicted Cytotoxicity')
    # Additional formatting code...
```

## Key Variables

- `dataset`: DataFrame containing cytotoxicity data for compounds at different concentrations
- `train`: DataFrame containing ChemBERTa embeddings for individual compounds
- `mixture_info`: DataFrame containing information about chemical mixtures
- `mix_emb_df`: DataFrame containing calculated embeddings for mixtures
- `X_train`, `y_train`: Training features and target values
- `X_test`, `y_test`: Testing features and target values
- `model`: The trained Random Forest Regressor model

## The `formula_descriptors` Module

The `formula_descriptors.py` module provides functions for calculating embeddings for chemical mixtures based on the embeddings of their constituent compounds and their mole fractions:

### Key Functions

1. `get_component_embedding(compound, mole_frac, embeddings)`: 
   - Retrieves the embedding for a given compound and scales it by the mole fraction

2. `calculate_mixture_embedding(compound_names, mole_fractions, embeddings)`: 
   - Calculates the mixture embedding for a set of compounds and mole fractions
   - Uses a weighted sum approach where each compound's embedding is weighted by its mole fraction

3. `formulate_descriptors(mixture_info, embeddings)`: 
   - Main function that formulates molecular descriptors for a given set of mixtures
   - Takes mixture information (compound names and mole fractions) and embeddings for individual compounds
   - Returns a DataFrame containing the embeddings for the mixtures

## Model Performance

The model achieves good performance in predicting cytotoxicity for both individual compounds and mixtures. The notebook includes visualizations comparing predicted and actual cytotoxicity values across different concentrations.

Performance metrics include:
- R² Score (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Conclusion

This notebook demonstrates a machine learning approach for predicting the cytotoxicity of chemical mixtures based on the properties of their constituent compounds. The model leverages ChemBERTa embeddings and a Random Forest Regressor to make accurate predictions, which could be valuable for toxicological risk assessment of chemical mixtures.