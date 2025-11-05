# RCDK.R Documentation

## Overview
RCDK.R is an R script designed for chemical informatics analysis. It processes chemical compounds represented as SMILES (Simplified Molecular Input Line Entry System) strings to generate molecular descriptors using the R Chemical Development Kit (RCDK). These descriptors can be used for various cheminformatics applications, particularly in toxicity prediction models.

## Purpose
The script serves to:
1. Import chemical compound data (names and SMILES representations)
2. Calculate molecular descriptors for each compound using RCDK
3. Clean and normalize the descriptor data
4. Combine the descriptors with compound information
5. Export the processed data for further analysis

## Required Packages
The script requires the following R packages:

- **rcdk**: R interface to the Chemical Development Kit, used for molecular descriptor calculation
- **readr**: For reading CSV files efficiently
- **mlbench**: Machine learning benchmark problems
- **caret**: Classification and Regression Training, for predictive modeling
- **dplyr**: Data manipulation functions
- **pROC**: Tools for visualizing, smoothing and comparing ROC curves

## Key Variables

| Variable | Description |
|----------|-------------|
| `Ingredient_Name_SMILES` | DataFrame containing compound names and their SMILES representations |
| `smiles` | Character vector of SMILES strings |
| `mols` | Molecular objects parsed from SMILES strings |
| `descriptor_names` | List of all available molecular descriptor names from RCDK |
| `molecular_descriptors` | Raw calculated molecular descriptors |
| `red_mol_desc` | Cleaned molecular descriptors (NAs and constant columns removed) |
| `mdnumrows` | Number of compounds (rows in the descriptor matrix) |
| `mdnumcol` | Number of descriptors (columns in the descriptor matrix) |
| `maxs` | Maximum values for each descriptor (used for normalization) |
| `mins` | Minimum values for each descriptor (used for normalization) |
| `scaled_molecular_descriptors` | Normalized molecular descriptors |
| `Data` | Combined data with compound names and normalized descriptors |
| `Data_df` | Final data frame ready for export |

## Workflow

1. **Data Import**: 
   - Reads compound data from "kim_single_ingredient.csv"

2. **Molecular Descriptor Calculation**:
   - Converts SMILES strings to molecular objects
   - Identifies all available descriptors
   - Calculates descriptors for each molecule

3. **Data Cleaning**:
   - Removes columns with NA values
   - Removes columns with constant values

4. **Data Normalization**:
   - Scales all descriptors to a 0-1 range

5. **Data Export**:
   - Combines compound names with normalized descriptors
   - Exports the final dataset to "KIM_rcdk_descriptors.csv"

## Usage

1. Ensure all required packages are installed:
   ```R
   install.packages(c("rcdk", "readr", "mlbench", "caret", "dplyr", "pROC"))
   ```

2. Prepare your input data:
   - Create a CSV file named "kim_single_ingredient.csv"
   - Include columns for compound names and SMILES strings
   - Ensure the SMILES column is named "smiles"

3. Run the script:
   ```R
   source("RCDK.R")
   ```

4. Access the results:
   - The processed data is available in the `Data_df` variable
   - The data is also exported to "KIM_rcdk_descriptors.csv"

## Notes

- The script uses `set.seed(42)` to ensure reproducibility
- Global variables are created using the `<<-` operator
- The script is designed for toxicity mixture analysis, as suggested by variable names

## Example Application

This script is typically used as a preprocessing step in toxicity prediction workflows, where molecular descriptors serve as features for machine learning models that predict compound toxicity or other biological activities.