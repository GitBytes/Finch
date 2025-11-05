# ChemBERTa Fine-tuning for Liver Toxicity Prediction

This folder contains resources for fine-tuning the ChemBERTa model to predict liver toxicity based on chemical compound data. It is part of a larger project focused on applying transformer-based models to chemical toxicity prediction.

## Contents

- **CSV Data Files**:
  - `AID_1224867_datatable_hepg2_24h.csv`: PubChem bioassay data for HepG2 cell line toxicity at 24 hours exposure
  - `AID_1224879_datatable_hepg2_40h.csv`: PubChem bioassay data for HepG2 cell line toxicity at 40 hours exposure

- **Python Scripts**:
  - `train_chemberta.py`: Script for fine-tuning ChemBERTa on liver toxicity data
  - `predict.py`: Script for making predictions with the fine-tuned model

- **Original Notebook** (for reference):
  - `finetune_demo.ipynb`: Original Jupyter notebook containing exploratory code

## Data Description

The CSV files contain extensive bioassay data from PubChem for chemical compounds tested on the HepG2 human liver cancer cell line at different exposure times (24h and 40h). Each file includes:

- SMILES representations of chemical compounds
- Activity outcomes (Active/Inactive) indicating toxicity
- Detailed dose-response data with multiple replicates
- Various metrics including potency, efficacy, and activity scores

## Model Information

This project uses the DeepChem/ChemBERTa-77M-MLM model, a BERT-based transformer pre-trained on molecular SMILES strings. The model is fine-tuned for binary classification to predict liver toxicity of chemical compounds.

## Implementation Details

The Python scripts implement:

1. Data loading and preprocessing of chemical compound data
2. Tokenization of SMILES strings using ChemBERTa's tokenizer
3. Dataset splitting into training and validation sets
4. Model fine-tuning with the Hugging Face Transformers library
5. Evaluation metrics including accuracy and binary cross-entropy
6. Prediction functionality for new chemical compounds

## Usage

### Training

To fine-tune the ChemBERTa model:

```bash
python train_chemberta.py --data_path AID_1224879_datatable_hepg2_40h.csv --epochs 200 --batch_size 64
```

Additional options:
```
--model_name: Pretrained model name (default: DeepChem/ChemBERTa-77M-MLM)
--output_dir: Directory to save the model (default: ./liver_results)
--learning_rate: Learning rate (default: 1e-5)
--warmup_steps: Number of warmup steps (default: 500)
--weight_decay: Weight decay (default: 0.01)
```

### Prediction

To make predictions with the fine-tuned model:

```bash
python predict.py --model_path ./liver_results/final --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" "C1=CC=C(C=C1)C(=O)O"
```

Or use a file containing SMILES:
```bash
python predict.py --model_path ./liver_results/final --input_file compounds.csv --smiles_column SMILES --output_file predictions.csv
```

## Model Training Parameters

- Learning rate: 1e-5
- Batch size: 64
- Training epochs: 200
- Optimizer: AdamW with weight decay of 0.01

This fine-tuned model can be used to predict the potential liver toxicity of new chemical compounds based on their SMILES representation.
