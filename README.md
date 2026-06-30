> [!IMPORTANT]
> This repository has been archived and is no longer maintained.
> The code is provided for historical reference and may contain unpatched or unknown vulnerabilities.
> It should not be used in production systems.

# ToxMix: Chemical Cytotoxicity Prediction Project

This repository contains code and resources for predicting the cytotoxicity of both individual chemical compounds and their mixtures using various machine learning approaches. The project leverages different molecular representation techniques and machine learning models to achieve accurate toxicity predictions.

## Repository Structure

### 1. ChemBERTa-pre_trained

This directory contains code for a machine learning model that predicts cytotoxicity using pre-trained ChemBERTa embeddings as molecular representations.

**Key Files:**
- `8020_positive-only.ipynb`: Jupyter notebook implementing a Random Forest Regression model for cytotoxicity prediction
- `formula_descriptors.py`: Module for calculating embeddings for chemical mixtures based on individual compound embeddings
- `80_20_level_model_pretrained.pkl`: Saved trained model
- `README_ChemBERTA-pre-trained.md`: Detailed documentation

**Data:**
- Chemical embeddings generated from ChemBERTa
- Cytotoxicity data for individual compounds and mixtures
- Mixture composition information

**Dependencies:**
- pandas, numpy, scikit-learn, matplotlib, pickle

**Approach:**
- Uses pre-trained ChemBERTa embeddings (384-dimensional) to represent chemical compounds
- Calculates mixture embeddings as weighted sums of individual compound embeddings
- Implements a Random Forest Regression model with hyperparameter tuning
- Evaluates performance using R², MAE, and RMSE metrics

### 2. RCDK

This directory contains code for a cytotoxicity prediction model using molecular descriptors generated from the R Chemistry Development Kit (RCDK).

**Key Files:**
- `formula_descriptors.py`: Module for calculating descriptors for chemical mixtures
- `README_RCDK.md`: Detailed documentation

**Dependencies:**
- pandas, numpy, scikit-learn, matplotlib, pickle

**Approach:**
- Uses RCDK-generated molecular descriptors to represent chemical compounds
- Calculates mixture descriptors as weighted sums of individual compound descriptors
- Implements a Random Forest Regression model with hyperparameter tuning
- Evaluates performance using R², MAE, and MSE metrics

### 3. Rcode

This directory contains R scripts for generating molecular descriptors using the R Chemical Development Kit.

**Key Files:**
- `RCDK.R`: R script for calculating molecular descriptors
- `kim_single_ingredient.csv`: Input data with compound names and SMILES strings
- `KIM_rcdk_descriptors.csv`: Output file with calculated descriptors
- `README_RCDK.md`: Detailed documentation

**Dependencies:**
- rcdk, readr, mlbench, caret, dplyr, pROC

**Functionality:**
- Imports chemical compound data (names and SMILES representations)
- Calculates molecular descriptors using RCDK
- Cleans and normalizes descriptor data
- Exports processed data for further analysis

### 4. chemberta_finetune_class

This directory contains resources for fine-tuning the ChemBERTa model specifically for liver toxicity prediction.

**Key Files:**
- `chemberta-liver.ipynb`: Notebook for fine-tuning ChemBERTa on liver toxicity data
- `AID_1224867_datatable_hepg2_24h.csv`: PubChem bioassay data for HepG2 cell line toxicity (24h exposure)
- `AID_1224879_datatable_hepg2_40h.csv`: PubChem bioassay data for HepG2 cell line toxicity (40h exposure)
- `README.md`: Detailed documentation

**Dependencies:**
- PyTorch, Transformers, scikit-learn, pandas

**Approach:**
- Uses DeepChem/ChemBERTa-77M-MLM model
- Fine-tunes for binary classification of liver toxicity
- Processes SMILES strings using ChemBERTa's tokenizer
- Evaluates using accuracy and binary cross-entropy metrics

### 5. ChemBERTa-PUBCHEM

This directory contains visualization outputs and possibly code related to ChemBERTa models trained on PubChem data.

**Contents:**
- Visualization figures for various chemical compounds and mixtures

### 6. pre_trained_embeddings

This directory likely contains pre-trained embeddings for chemical compounds, used as inputs for the machine learning models.

## Project Overview

The ToxMix project aims to develop accurate predictive models for chemical toxicity, with a particular focus on predicting the cytotoxicity of chemical mixtures. The project employs multiple approaches:

1. **ChemBERTa Embeddings**: Using transformer-based molecular representations from pre-trained ChemBERTa models
2. **RCDK Descriptors**: Using traditional molecular descriptors calculated with the R Chemistry Development Kit
3. **Fine-tuned ChemBERTa**: Specifically fine-tuning ChemBERTa for liver toxicity prediction

A key innovation in this project is the methodology for representing chemical mixtures, which uses a weighted sum approach based on the mole fractions of constituent compounds.

## Usage

Each subdirectory contains its own README with specific instructions for using the code and models in that directory. In general, the workflow involves:

1. Generating molecular representations (either ChemBERTa embeddings or RCDK descriptors)
2. Training machine learning models (typically Random Forest Regression)
3. Evaluating model performance on test data
4. Using trained models to predict toxicity of new compounds or mixtures

## Dependencies

The project uses a combination of Python and R libraries:

**Python:**
- pandas, numpy, scikit-learn, matplotlib, pickle
- PyTorch, Transformers (for ChemBERTa fine-tuning)

**R:**
- rcdk, readr, mlbench, caret, dplyr, pROC

## Data

The project uses several types of data:
- Chemical structures represented as SMILES strings
- Molecular descriptors and embeddings
- Cytotoxicity measurements for individual compounds and mixtures
- Mixture composition information (compound names and mole fractions)

## Model Performance

The models achieve good performance in predicting cytotoxicity for both individual compounds and mixtures, as demonstrated by various evaluation metrics and visualization plots included in the respective directories.