"""
Data Preprocessing Module

This module handles the critical first step in the ML pipeline: data preparation.

Why preprocessing matters:
- Stock market data comes from different sources and may have inconsistencies
- Missing dates or values can break time-series models
- Duplicate entries need to be identified and removed
- Both datasets must be aligned on the same dates for valid predictions

Key decisions made:
1. Use inner join to keep only dates present in both datasets (conservative approach)
2. Forward-fill then backward-fill missing values (preserves temporal patterns)
3. Remove duplicates keeping first occurrence (assumes earlier data is more reliable)
4. Automatic column detection to handle different CSV formats flexibly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings


def load_data(data_path: str, stock_price_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the data and stock price datasets from CSV files.
    
    This function handles the initial data loading and performs basic validation
    to ensure the datasets are in the expected format. It automatically converts
    date strings to datetime objects for proper temporal handling.
    
    Args:
        data_path: Path to the data.csv file containing independent variables
        stock_price_path: Path to the stock_price.csv file containing target prices
        
    Returns:
        Tuple of (data_df, stock_price_df) as pandas DataFrames
        
    Raises:
        FileNotFoundError: If either CSV file doesn't exist
        ValueError: If Date column is missing or cannot be parsed
    """
    print("Loading datasets...")
    
    # Validate file paths exist
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not Path(stock_price_path).exists():
        raise FileNotFoundError(f"Stock price file not found: {stock_price_path}")
    
    # Load CSV files
    try:
        data_df = pd.read_csv(data_path)
        stock_price_df = pd.read_csv(stock_price_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")
    
    # Validate Date column exists in both datasets
    if 'Date' not in data_df.columns:
        raise ValueError(f"'Date' column not found in {data_path}. Available columns: {list(data_df.columns)}")
    if 'Date' not in stock_price_df.columns:
        raise ValueError(f"'Date' column not found in {stock_price_path}. Available columns: {list(stock_price_df.columns)}")
    
    # Convert Date column to datetime for proper temporal operations
    # This is crucial because string dates don't sort correctly (e.g., "2024-10-1" > "2024-9-30")
    try:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}. Ensure dates are in a recognizable format (e.g., YYYY-MM-DD)")
    
    # Log loaded data information
    print(f"✓ Data dataset loaded: {data_df.shape[0]} rows, {data_df.shape[1]} columns")
    print(f"✓ Stock price dataset loaded: {stock_price_df.shape[0]} rows, {stock_price_df.shape[1]} columns")
    
    # Automatically detect feature columns (all non-Date columns)
    data_features = [col for col in data_df.columns if col != 'Date']
    price_features = [col for col in stock_price_df.columns if col != 'Date']
    print(f"  Features in data.csv: {data_features}")
    print(f"  Features in stock_price.csv: {price_features}")
    
    # Warn if we have only one feature (limited predictive power)
    if len(data_features) == 1:
        warnings.warn(f"⚠ Only 1 feature found ({data_features[0]}). Single-feature models have limited predictive power.", UserWarning)
    
    return data_df, stock_price_df


def clean_data(data_df: pd.DataFrame, stock_price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the datasets by handling missing values and removing duplicates.
    
    Real-world data is messy. This function addresses common data quality issues:
    
    Missing values handling:
    - Forward fill (ffill): Assumes last known value persists (common in financial data)
    - Backward fill (bfill): Fills any remaining NaNs at the start with next known value
    - This is better than dropping rows, which would create gaps in the time series
    
    Duplicate handling:
    - Keep='first' assumes earlier entries are more reliable (from primary source)
    - Duplicates can occur from data collection errors or multiple updates
    
    Args:
        data_df: Data features dataframe
        stock_price_df: Stock price dataframe
        
    Returns:
        Tuple of cleaned (data_df, stock_price_df)
    """
    print("\nCleaning datasets...")
    
    # Check for missing values before cleaning
    data_missing = data_df.isnull().sum().sum()
    price_missing = stock_price_df.isnull().sum().sum()
    print(f"Missing values in data: {data_missing}")
    print(f"Missing values in stock price: {price_missing}")
    
    if data_missing > 0 or price_missing > 0:
        warnings.warn(f"⚠ Found missing values. Applying forward-fill + backward-fill strategy.", UserWarning)
    
    # Remove duplicates based on Date (keep first occurrence)
    # Why 'first'? Assumes primary data source is logged first, later entries are duplicates
    data_duplicates = data_df.duplicated(subset=['Date']).sum()
    price_duplicates = stock_price_df.duplicated(subset=['Date']).sum()
    
    if data_duplicates > 0:
        print(f"⚠ Removing {data_duplicates} duplicate dates from data.csv")
    if price_duplicates > 0:
        print(f"⚠ Removing {price_duplicates} duplicate dates from stock_price.csv")
    
    data_df = data_df.drop_duplicates(subset=['Date'], keep='first')
    stock_price_df = stock_price_df.drop_duplicates(subset=['Date'], keep='first')
    
    # Fill missing values with forward fill (ffill) then backward fill (bfill)
    # This preserves temporal patterns better than mean/median imputation
    data_df = data_df.ffill().bfill()
    stock_price_df = stock_price_df.ffill().bfill()
    
    # Verify no missing values remain
    remaining_data_missing = data_df.isnull().sum().sum()
    remaining_price_missing = stock_price_df.isnull().sum().sum()
    
    if remaining_data_missing > 0 or remaining_price_missing > 0:
        warnings.warn(f"⚠ Still have missing values after cleaning! Data: {remaining_data_missing}, Price: {remaining_price_missing}", UserWarning)
    
    print(f"✓ After cleaning - Data shape: {data_df.shape}, Stock price shape: {stock_price_df.shape}")
    
    return data_df, stock_price_df


def sort_and_align(data_df: pd.DataFrame, stock_price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort both datasets by date and merge them on the Date column.
    
    Critical decisions for time-series data:
    
    1. Sorting by date: Essential for time-series models. Chronological order ensures
       that features at time t can only predict outcomes at t+1 or later (no data leakage).
    
    2. Inner join strategy: Only keeps dates present in BOTH datasets. This is conservative
       but ensures we never make predictions with incomplete information.
       - Alternative (outer join): Would keep all dates but introduce more missing values
       - Why inner? Better to have less data that's complete than more data that's partial
    
    3. Automatic column renaming: Detects the price column name dynamically and standardizes
       it to 'Stock_Price' for consistent downstream processing.
    
    Args:
        data_df: Data features dataframe (already cleaned and sorted)
        stock_price_df: Stock price dataframe (already cleaned and sorted)
        
    Returns:
        Merged dataframe sorted by date with standardized column names
    """
    print("\nSorting and aligning datasets...")
    
    # Sort by date to ensure chronological order
    # This is crucial: models trained on unsorted data can "see the future" and give misleadingly good results
    data_df = data_df.sort_values('Date').reset_index(drop=True)
    stock_price_df = stock_price_df.sort_values('Date').reset_index(drop=True)
    
    # Automatically detect the price column (any column that's not 'Date')
    price_col = [col for col in stock_price_df.columns if col != 'Date'][0]
    
    # Standardize column name for consistency across the pipeline
    if price_col != 'Stock_Price':
        print(f"  Renaming '{price_col}' to 'Stock_Price' for standardization")
        stock_price_df = stock_price_df.rename(columns={price_col: 'Stock_Price'})
    
    # Merge datasets on Date using inner join
    # Inner join = only keep dates present in BOTH datasets (conservative but complete)
    merged_df = pd.merge(data_df, stock_price_df, on='Date', how='inner')
    
    # Calculate and report data loss from merge
    data_loss = len(data_df) - len(merged_df)
    price_loss = len(stock_price_df) - len(merged_df)
    
    if data_loss > 0 or price_loss > 0:
        print(f"⚠ Inner join removed {data_loss} dates from data.csv and {price_loss} dates from stock_price.csv")
        print(f"  (These dates existed in one dataset but not the other)")
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    print(f"  Date range: {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {list(merged_df.columns)}")
    
    # Validate we have enough data for meaningful modeling
    if len(merged_df) < 100:
        warnings.warn(f"⚠ Only {len(merged_df)} samples available. Need at least 100 for reliable model training.", UserWarning)
    
    return merged_df


def preprocess_pipeline(data_path: str, stock_price_path: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline that orchestrates all cleaning steps.
    
    This is the entry point for data preparation. It runs all preprocessing
    steps in the correct order to produce a clean, aligned dataset ready
    for feature engineering.
    
    Pipeline steps:
    1. Load: Read CSV files and validate structure
    2. Clean: Handle missing values and remove duplicates
    3. Align: Sort by date and merge on common dates
    
    Args:
        data_path: Path to the data.csv file containing independent variables
        stock_price_path: Path to the stock_price.csv file containing target prices
        
    Returns:
        Preprocessed and merged dataframe ready for feature engineering
    """
    # Step 1: Load both datasets
    data_df, stock_price_df = load_data(data_path, stock_price_path)
    
    # Step 2: Clean both datasets
    data_df, stock_price_df = clean_data(data_df, stock_price_df)
    
    # Step 3: Sort chronologically and merge on Date
    merged_df = sort_and_align(data_df, stock_price_df)
    
    print(f"\n✓ Preprocessing complete! Final dataset has {len(merged_df)} rows")
    
    return merged_df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_path = "../data/data.csv"
    stock_price_path = "../data/stock_price.csv"
    
    df = preprocess_pipeline(data_path, stock_price_path)
    print("\nFirst few rows of merged dataset:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
