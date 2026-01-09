"""
Main Execution Script for Stock Price Prediction

This script orchestrates the complete machine learning pipeline:
1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Evaluation

Run this script to execute the entire pipeline end-to-end.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocess import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.train import training_pipeline
from src.evaluate import evaluation_pipeline


def main():
    """
    Main function to run the complete ML pipeline.
    """
    print("="*80)
    print(" " * 20 + "STOCK PRICE PREDICTION")
    print(" " * 15 + "Machine Learning Pipeline")
    print("="*80)
    
    # Define paths
    data_path = "data/data.csv"
    stock_price_path = "data/stock_price.csv"
    
    try:
        # Step 1: Preprocessing
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        merged_df = preprocess_pipeline(data_path, stock_price_path)
        
        # Step 2: Feature Engineering
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        df_full, X, y, feature_names = feature_engineering_pipeline(merged_df)
        
        # Step 3: Model Training
        print("\n" + "="*80)
        print("STEP 3: MODEL TRAINING")
        print("="*80)
        results = training_pipeline(X, y, feature_names, test_size=0.2)
        
        # Step 4: Model Evaluation
        print("\n" + "="*80)
        print("STEP 4: MODEL EVALUATION")
        print("="*80)
        evaluation_pipeline(results)
        
        # Final Summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nOutputs saved:")
        print("  - Models: models/")
        print("  - Visualizations: outputs/")
        print("\nCheck the outputs/ directory for prediction plots and analysis.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Pipeline execution failed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
