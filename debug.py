import numpy as np
import os
import sys
from pathlib import Path

def analyze_fid_features(file_path):
    """Analyze the FID features stored in an NPZ file."""
    print(f"\nAnalyzing FID features from: {file_path}")
    print("-" * 80)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return
    
    # Load the NPZ file
    try:
        data = np.load(file_path)
        print("\nAvailable arrays in the NPZ file:")
        print(list(data.keys()))
        
        # Analyze the 'act' array
        if 'act' in data:
            acts = data['act']
            print("\nActivation array analysis:")
            print(f"Shape: {acts.shape}")
            print(f"Data type: {acts.dtype}")
            print(f"Total elements: {acts.size}")
            print(f"Memory usage: {acts.nbytes / (1024*1024):.2f} MB")
            
            print("\nNumeric statistics:")
            print(f"Mean: {acts.mean():.6f}")
            print(f"Std: {acts.std():.6f}")
            print(f"Min: {acts.min():.6f}")
            print(f"Max: {acts.max():.6f}")
            
            # Check for NaN or Inf values
            print("\nData quality checks:")
            print(f"Contains NaN: {np.isnan(acts).any()}")
            print(f"Contains Inf: {np.isinf(acts).any()}")
            
            # Sample a few values
            print("\nSample values (first 5 rows, first 5 columns):")
            print(acts[:5, :5])
            
        else:
            print("\nWarning: No 'act' array found in the file!")
            
        # Check if there are other arrays and their sizes
        print("\nAll arrays in the file:")
        for key in data.keys():
            arr = data[key]
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
            
    except Exception as e:
        print(f"Error loading file: {str(e)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # File path
    file_path = "ckpt_gcs/tokenizer/acts/val_256_act.npz"
    
    try:
        analyze_fid_features(file_path)
    except Exception as e:
        print(f"Error: {str(e)}")