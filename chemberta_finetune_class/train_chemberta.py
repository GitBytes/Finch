#!/usr/bin/env python3
"""
ChemBERTa Fine-tuning for Liver Toxicity Prediction
This script fine-tunes the ChemBERTa model on liver toxicity data.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Define the dataset class
class ChemicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'binary_cross_entropy': log_loss(labels, logits)
    }

def load_and_preprocess_data(data_path, skiprows=range(1, 6)):
    """Load and preprocess the chemical compound data."""
    # Load data
    data = pd.read_csv(data_path, skiprows=skiprows)
    data = data.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"])
    
    # Create binary classification labels
    data['Class'] = data['PUBCHEM_ACTIVITY_OUTCOME'].apply(lambda x: 1 if x == 'Active' else 0)
    
    return data

def prepare_datasets(data, tokenizer, test_size=0.2, random_state=42):
    """Split data and prepare datasets for training and validation."""
    # Split data into train and validation sets
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        data.PUBCHEM_EXT_DATASOURCE_SMILES.values,
        data.Class.values,
        test_size=test_size,
        stratify=data.Class.values,
        random_state=random_state
    )
    
    # Tokenize SMILES strings
    train_encodings = tokenizer(
        list(train_smiles),
        truncation=True,
        padding=True,
        max_length=512
    )
    
    val_encodings = tokenizer(
        list(val_smiles),
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Create dataset objects
    train_dataset = ChemicalDataset(train_encodings, train_labels)
    val_dataset = ChemicalDataset(val_encodings, val_labels)
    
    return train_dataset, val_dataset

def main(args):
    # Load and preprocess data
    data = load_and_preprocess_data(args.data_path)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(data, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        evaluation_strategy="steps",
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"{args.output_dir}/final")
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ChemBERTa for liver toxicity prediction")
    
    parser.add_argument("--data_path", type=str, default="AID_1224879_datatable_hepg2_40h.csv",
                        help="Path to the data CSV file")
    parser.add_argument("--model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM",
                        help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./liver_results",
                        help="Directory to save the model")
    parser.add_argument("--logging_dir", type=str, default="./liver_logs",
                        help="Directory for logs")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    
    args = parser.parse_args()
    main(args)
