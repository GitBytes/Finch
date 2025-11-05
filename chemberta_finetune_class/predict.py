#!/usr/bin/env python3
"""
ChemBERTa Prediction Script for Liver Toxicity
This script uses a fine-tuned ChemBERTa model to predict liver toxicity of chemical compounds.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    return model, tokenizer

def predict_toxicity(model, tokenizer, smiles_list):
    """Predict toxicity for a list of SMILES strings."""
    # Tokenize SMILES
    inputs = tokenizer(
        smiles_list,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    # Convert to numpy for easier handling
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Create results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'SMILES': smiles_list[i],
            'Prediction': 'Toxic' if pred == 1 else 'Non-toxic',
            'Confidence': prob[pred]
        })
    
    return results

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Load SMILES from file or use the provided ones
    if args.input_file:
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            smiles_list = df[args.smiles_column].tolist()
        else:
            with open(args.input_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
    else:
        smiles_list = args.smiles
    
    # Make predictions
    results = predict_toxicity(model, tokenizer, smiles_list)
    
    # Display results
    for result in results:
        print(f"SMILES: {result['SMILES']}")
        print(f"Prediction: {result['Prediction']}")
        print(f"Confidence: {result['Confidence']:.4f}")
        print("-" * 50)
    
    # Save results if output file is specified
    if args.output_file:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict liver toxicity using fine-tuned ChemBERTa")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    
    # Input options - either file or direct SMILES
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str,
                            help="Path to file containing SMILES strings (one per line or CSV)")
    input_group.add_argument("--smiles", type=str, nargs='+',
                            help="One or more SMILES strings to predict")
    
    # Additional options
    parser.add_argument("--smiles_column", type=str, default="SMILES",
                        help="Column name containing SMILES in CSV input (default: SMILES)")
    parser.add_argument("--output_file", type=str,
                        help="Path to save prediction results as CSV")
    
    args = parser.parse_args()
    main(args)
