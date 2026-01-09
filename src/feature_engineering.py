"""
Feature Engineering Module

This module transforms raw data into features that models can learn from.

THE CORE INSIGHT:
Stock prices don't exist in isolation - they react to CHANGES in market conditions.
If volume suddenly spikes, prices often move. If it stays flat, prices may be stable.

WHY DAY-OVER-DAY CHANGES?
1. Raw values (e.g., "volume = 5000") don't tell us much across different time periods
2. CHANGES (e.g., "volume increased by 500") capture market dynamics
3. This is similar to derivatives in calculus - we care about the rate of change

TEMPORAL ALIGNMENT (CRITICAL):
- Features at time t represent changes from t-1 to t (what happened during the day)
- Target at time t is the price at t+1 (what we want to predict for tomorrow)
- This alignment ensures we're only using past information to predict the future

MATHEMATICAL NOTATION:
Let X(t) be any feature value at time t
- Feature: ΔX(t) = X(t) - X(t-1)  [today's change from yesterday]
- Target: Price(t+1)               [tomorrow's price]
- Model learns: Price(t+1) = f(ΔX(t))  [tomorrow's price as function of today's changes]

EXAMPLE:
If on Monday we see volume increased by 1000, sentiment improved by 0.5,
we use those changes to predict Tuesday's stock price.
"""

import pandas as pd
import numpy as np
import warnings


def compute_day_over_day_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute day-over-day changes for all independent variables.
    
    MATHEMATICAL FORMULA:
    For any feature X at time t:
        ΔX(t) = X(t) - X(t-1)
    
    WHY THIS WORKS:
    - pandas.diff() computes: value[i] - value[i-1] for each row
    - First row becomes NaN (no previous day) and is dropped
    - Resulting changes capture daily market movements
    
    EXAMPLE:
    If volume on Mon=1000, Tue=1200, Wed=1100:
        Mon_change = NaN (no previous day)
        Tue_change = 1200 - 1000 = +200 (volume increased)
        Wed_change = 1100 - 1200 = -100 (volume decreased)
    
    VALIDATION:
    - Checks that changes are computed for all features
    - Warns if changes are suspiciously large (data quality issue)
    - Logs statistics to verify reasonableness
    
    Args:
        df: Merged dataframe with Date, features, and Stock_Price
        
    Returns:
        Dataframe with day-over-day change features
    """
    print("\nComputing day-over-day changes...")
    
    # Create a copy to avoid modifying the original dataframe
    df_changes = df.copy()
    
    # Identify feature columns (exclude Date and Stock_Price which are not features)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Stock_Price']]
    
    print(f"  Computing changes for features: {feature_cols}")
    
    # Compute day-over-day changes for each feature using pandas diff()
    # diff() calculates: value[i] - value[i-1]
    for col in feature_cols:
        # Calculate change: ΔX(t) = X(t) - X(t-1)
        df_changes[f'{col}_change'] = df[col].diff()
        
        # Sanity check: are changes reasonable?
        change_col = f'{col}_change'
        mean_change = df_changes[change_col].mean()
        std_change = df_changes[change_col].std()
        max_change = df_changes[change_col].max()
        min_change = df_changes[change_col].min()
        
        # Log statistics for transparency
        print(f"    {col}_change: mean={mean_change:.4f}, std={std_change:.4f}, range=[{min_change:.4f}, {max_change:.4f}]")
        
        # Warn if changes are suspiciously large (potential data quality issue)
        if abs(max_change) > 10 * std_change or abs(min_change) > 10 * std_change:
            warnings.warn(f"⚠ {col}_change has extreme values (>10 std devs). Check data quality!", UserWarning)
    
    # The first row will have NaN values for changes (no previous day to compare)
    # We drop it because we can't compute changes without a baseline
    rows_before = len(df_changes)
    df_changes = df_changes.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df_changes)
    
    print(f"  ✓ After computing changes, shape: {df_changes.shape}")
    print(f"    Dropped {rows_dropped} row(s) due to NaN values (expected: first row has no previous day)")
    
    return df_changes


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable: next day's stock price.
    
    TEMPORAL ALIGNMENT EXPLAINED:
    We need to align "what we know today" with "what we want to predict tomorrow".
    
    Current data structure:
        Date        | Feature_change | Stock_Price
        2024-01-01  | ΔX(0→1)       | Price(1)
        2024-01-02  | ΔX(1→2)       | Price(2)
        2024-01-03  | ΔX(2→3)       | Price(3)
    
    Goal: Use changes at day t to predict price at day t+1
        Features(t)  →  Target(t+1)
        ΔX(0→1)      →  Price(2)
        ΔX(1→2)      →  Price(3)
    
    IMPLEMENTATION:
    pandas shift(-1) moves values UP one row:
        next_day_price = [Price(2), Price(3), NaN]
    
    After this, row t contains:
        - Features: Changes from t-1 to t (what happened today)
        - Target: Price at t+1 (what we want to predict for tomorrow)
    
    CRITICAL INSIGHT:
    This ensures the model ONLY sees past information (no data leakage).
    We never use tomorrow's information to predict tomorrow's price.
    
    Args:
        df: Dataframe with features and Stock_Price
        
    Returns:
        Dataframe with aligned features and next_day_price target
    """
    print("\nCreating target variable (next day's stock price)...")
    
    df_target = df.copy()
    
    # Shift stock price backwards by 1 position to align with today's features
    # This creates: features(today) → price(tomorrow)
    df_target['next_day_price'] = df_target['Stock_Price'].shift(-1)
    
    # Validate the alignment is correct
    # Check: next_day_price should be equal to tomorrow's Stock_Price
    sample_idx = min(5, len(df_target) - 2)  # Check a middle row
    if sample_idx > 0:
        current_stock = df_target.loc[sample_idx, 'Stock_Price']
        next_target = df_target.loc[sample_idx, 'next_day_price']
        following_stock = df_target.loc[sample_idx + 1, 'Stock_Price']
        
        if abs(next_target - following_stock) > 0.01:  # Allow small floating point errors
            warnings.warn(f"⚠ Target alignment may be incorrect! next_day_price doesn't match next Stock_Price", UserWarning)
    
    # Drop the last row (no next day price available - we can't predict beyond our data)
    rows_before = len(df_target)
    df_target = df_target.dropna(subset=['next_day_price']).reset_index(drop=True)
    rows_dropped = rows_before - len(df_target)
    
    print(f"  ✓ After creating target, shape: {df_target.shape}")
    print(f"    Target variable: next_day_price (tomorrow's stock price)")
    print(f"    Dropped {rows_dropped} row(s) (last date has no tomorrow to predict)")
    
    # Log target statistics for transparency
    print(f"    Target statistics: mean=${df_target['next_day_price'].mean():.2f}, "
          f"std=${df_target['next_day_price'].std():.2f}, "
          f"range=[${df_target['next_day_price'].min():.2f}, ${df_target['next_day_price'].max():.2f}]")
    
    return df_target


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepare final feature matrix (X) and target vector (y) for model training.
    
    WHY ONLY USE CHANGE FEATURES:
    The original raw features (volume, sentiment, etc.) are still in the dataframe,
    but we explicitly select ONLY the _change columns for modeling. Here's why:
    
    1. Raw values are non-stationary (trends, seasonality) - hard to model
    2. Changes are more stationary (fluctuate around zero) - easier to model
    3. Changes capture the "signal" we care about: market dynamics
    4. Using both raw and changes would introduce multicollinearity
    
    SANITY CHECKS:
    - Ensure we actually have change features (if not, something went wrong)
    - Validate X and y have matching dimensions
    - Warn if we have very few features (limited predictive power)
    
    Args:
        df: Dataframe with change features and next_day_price
        
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature matrix (DataFrame) containing only day-over-day changes
        - y: Target vector (Series) containing next day stock prices
        - feature_names: List of feature column names for later reference
    """
    print("\nPreparing features and target...")
    
    # Select only the change features (columns ending with '_change')
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Sanity check: do we have any change features?
    if len(change_cols) == 0:
        raise ValueError("❌ No change features found! Something went wrong in compute_day_over_day_changes()")
    
    # Extract feature matrix (X) and target vector (y)
    X = df[change_cols]
    y = df['next_day_price']
    
    # Validate dimensions match
    if len(X) != len(y):
        raise ValueError(f"❌ Dimension mismatch! X has {len(X)} samples but y has {len(y)} samples")
    
    # Validate no missing values in features or target
    if X.isnull().sum().sum() > 0:
        warnings.warn(f"⚠ Features (X) contain {X.isnull().sum().sum()} missing values!", UserWarning)
    if y.isnull().sum() > 0:
        warnings.warn(f"⚠ Target (y) contains {y.isnull().sum()} missing values!", UserWarning)
    
    # Check feature quality: are there any features with zero variance?
    zero_var_features = []
    for col in change_cols:
        if X[col].std() < 1e-10:  # Essentially zero variance
            zero_var_features.append(col)
    
    if zero_var_features:
        warnings.warn(f"⚠ Features with zero variance (won't help prediction): {zero_var_features}", UserWarning)
    
    # Log final feature and target info
    print(f"  ✓ Feature matrix shape: {X.shape} (rows=samples, columns=features)")
    print(f"  ✓ Target vector shape: {y.shape}")
    print(f"  ✓ Features used: {change_cols}")
    
    # Warn if we have very few features (limited predictive power)
    if len(change_cols) == 1:
        warnings.warn(f"⚠ Only 1 feature available ({change_cols[0]}). Single-feature models have limited accuracy.", UserWarning)
    elif len(change_cols) < 3:
        warnings.warn(f"⚠ Only {len(change_cols)} features available. More features generally improve predictions.", UserWarning)
    
    return X, y, change_cols


def feature_engineering_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    """
    Complete feature engineering pipeline that transforms preprocessed data into ML-ready features.
    
    This function orchestrates all feature engineering steps in the correct order:
    1. Compute day-over-day changes for all features (ΔX)
    2. Create target variable (next day's price) with proper temporal alignment
    3. Extract feature matrix (X) and target vector (y)
    
    The result is data ready for model training with guaranteed:
    - No data leakage (features at t only predict target at t+1)
    - No missing values
    - Proper temporal ordering
    
    Args:
        df: Preprocessed and merged dataframe from preprocess.py
        
    Returns:
        Tuple of (df_full, X, y, feature_names) where:
        - df_full: Full dataframe with all columns (useful for analysis)
        - X: Feature matrix ready for sklearn models
        - y: Target vector ready for sklearn models
        - feature_names: List of feature names (for interpretation later)
    """
    # Step 1: Compute day-over-day changes (ΔX = X(t) - X(t-1))
    df_changes = compute_day_over_day_changes(df)
    
    # Step 2: Create target variable with proper temporal alignment
    # Align: features(today) → price(tomorrow)
    df_target = create_target_variable(df_changes)
    
    # Step 3: Extract feature matrix (X) and target vector (y)
    X, y, feature_names = prepare_features_and_target(df_target)
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*80}")
    print(f"  Final dataset: {len(df_target)} samples")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")
    print(f"  Target: next_day_price (tomorrow's stock price)")
    print(f"  Ready for model training!")
    print(f"{'='*80}\n")
    
    return df_target, X, y, feature_names


if __name__ == "__main__":
    # Test the feature engineering pipeline
    from preprocess import preprocess_pipeline
    
    data_path = "../data/data.csv"
    stock_price_path = "../data/stock_price.csv"
    
    # Preprocess
    df = preprocess_pipeline(data_path, stock_price_path)
    
    # Feature engineering
    df_full, X, y, feature_names = feature_engineering_pipeline(df)
    
    print("\nSample of features (X):")
    print(X.head())
    print("\nSample of target (y):")
    print(y.head())
    print("\nFeature statistics:")
    print(X.describe())
