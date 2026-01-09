"""
Model Training Module

This module trains machine learning models to predict next-day stock prices.

MODEL SELECTION RATIONALE:

1. Linear Regression:
   - Acts as a baseline - if complex models don't beat this, something's wrong
   - Interpretable: coefficients tell us HOW features affect prices
   - Fast to train and predict
   - Assumption: linear relationship between feature changes and price changes
   - Formula: Price(t+1) = β₀ + β₁·ΔFeature₁(t) + β₂·ΔFeature₂(t) + ...

2. Random Forest:
   - Captures non-linear patterns (e.g., "volume spike + sentiment drop = big price move")
   - Robust to outliers and noisy data
   - Provides feature importances (which features matter most)
   - Ensemble method: averages predictions from many decision trees
   - Less interpretable than linear regression, but often more accurate

KEY PRINCIPLES:

1. NO SHUFFLING: Time series data must maintain temporal order
   - Shuffling would let the model "see the future" during training
   - We use the first 80% for training, last 20% for testing
   - This mimics real-world scenario: train on past, predict future

2. FEATURE SCALING: StandardScaler normalizes features to mean=0, std=1
   - Why? Different features have different ranges (volume in millions, sentiment 0-1)
   - Scaling ensures no feature dominates just because of its scale
   - Fit scaler on training data ONLY to avoid data leakage

3. REPRODUCIBILITY: Set random_state for consistent results
   - Random Forest uses random sampling of features/data
   - Without random_state, results vary each run
   - With random_state=42, results are identical every time
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import warnings


def train_test_split_temporal(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Split data into train and test sets WITHOUT shuffling (time series rule).
    
    WHY NO SHUFFLING?
    Imagine training on data from March and testing on January. The model would
    learn patterns from the "future" (March) to predict the "past" (January).
    This gives misleadingly good results that don't work in real trading.
    
    CORRECT APPROACH:
    - Train: First 80% of chronological data (old history)
    - Test: Last 20% of chronological data (recent history)
    - This mimics reality: learn from past, predict future
    
    EXAMPLE:
    If we have data from Jan 1 to Dec 31 (365 days):
    - Training: Jan 1 to Oct 14 (292 days)
    - Testing: Oct 15 to Dec 31 (73 days)
    
    The model learns from Jan-Oct patterns and is evaluated on Oct-Dec patterns.
    
    Args:
        X: Feature matrix (already sorted chronologically)
        y: Target vector (already sorted chronologically)
        test_size: Proportion of data to use for testing (default 0.2 = 20%)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    print("⚠ Maintaining temporal order (NO shuffling) - critical for time series!")
    
    # Calculate split index
    # Example: if len(X) = 1000 and test_size = 0.2, split_idx = 800
    split_idx = int(len(X) * (1 - test_size))
    
    # Split: everything before split_idx goes to training, after goes to testing
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Sanity check: ensure we have enough data in both sets
    if len(X_train) < 50:
        warnings.warn(f"⚠ Training set very small ({len(X_train)} samples). Need at least 50 for reliable training.", UserWarning)
    if len(X_test) < 20:
        warnings.warn(f"⚠ Test set very small ({len(X_test)} samples). Need at least 20 for reliable evaluation.", UserWarning)
    
    print(f"  ✓ Training set size: {len(X_train)} samples")
    print(f"  ✓ Test set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler (mean=0, std=1).
    
    WHY SCALING MATTERS:
    Imagine two features:
    - Volume change: ranges from -1,000,000 to +1,000,000
    - Sentiment change: ranges from -1 to +1
    
    Without scaling, the model thinks volume is "1 million times more important"
    just because the numbers are bigger. Scaling makes them comparable.
    
    STANDARDIZATION FORMULA:
    For each feature: scaled_value = (value - mean) / std_deviation
    
    Result: All features have mean=0, std=1, making them directly comparable.
    
    CRITICAL: Fit scaler on training data ONLY!
    - Scaler calculates mean/std from training data
    - Apply same transformation to test data
    - If we fit on test data, we "leak" information about the future
    
    Args:
        X_train: Training features (not yet scaled)
        X_test: Test features (not yet scaled)
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
        Scaler is returned so we can scale future data the same way
    """
    print("\nScaling features using StandardScaler...")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data (learns mean and std from training set)
    # Then transform training data using those statistics
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using SAME mean/std learned from training
    # This is correct: we only scale, we don't re-learn statistics
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names (helpful for debugging)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Log scaling statistics for transparency
    print("  ✓ Features scaled successfully")
    print(f"    Learned means: {scaler.mean_}")
    print(f"    Learned std devs: {scaler.scale_}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    WHY LINEAR REGRESSION?
    It's the simplest model - a baseline to compare against. If Random Forest
    doesn't beat this, something is wrong with our features or data.
    
    WHAT IT DOES:
    Finds the best straight-line relationship between features and target.
    Formula: next_day_price = intercept + (coef1 * feature1_change) + (coef2 * feature2_change) + ...
    
    INTERPRETATION OF COEFFICIENTS:
    If Volume_change has coefficient +0.05:
    - "A 1-unit increase in daily volume change leads to $0.05 increase in next day's price"
    
    ADVANTAGES:
    - Fast to train (milliseconds even on large datasets)
    - Interpretable: coefficients show feature importance and direction
    - Transparent: we can explain predictions to stakeholders
    
    LIMITATIONS:
    - Assumes linear relationships (real markets are more complex)
    - Can't capture interactions (e.g., "high volume AND negative sentiment")
    - Sensitive to outliers
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training target (stock prices)
        
    Returns:
        Trained Linear Regression model
    """
    print("\n" + "="*80)
    print("TRAINING: Linear Regression (Baseline Model)")
    print("="*80)
    
    # Initialize and train the model
    # LinearRegression uses Ordinary Least Squares: minimizes sum of squared errors
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Log model details
    print("  ✓ Training complete!")
    print(f"    Number of features: {len(model.coef_)}")
    print(f"    Intercept (baseline price): ${model.intercept_:.2f}")
    print(f"    Equation: Price = ${model.intercept_:.2f} + " + 
          " + ".join([f"({coef:.4f} * {feat})" for feat, coef in zip(X_train.columns, model.coef_)][:2]) + " ...")
    
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                        n_estimators: int = 100, max_depth: int = 10,
                        random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    WHY RANDOM FOREST?
    Markets are complex and non-linear. Random Forest can capture patterns like:
    - "When volume spikes AND sentiment drops, prices usually fall hard"
    - "Small volume changes don't matter much, but large ones do"
    
    HOW IT WORKS:
    1. Grows many decision trees (n_estimators = how many)
    2. Each tree is trained on a random subset of data and features
    3. Final prediction = average of all tree predictions
    4. This "ensemble" approach reduces overfitting and improves accuracy
    
    KEY HYPERPARAMETERS:
    - n_estimators (100): More trees = more stable predictions, but slower
      - Too few: underfitting (misses patterns)
      - Too many: diminishing returns (computation cost)
    - max_depth (10): How deep each tree can grow
      - Too shallow: can't learn complex patterns
      - Too deep: overfits to training noise
    - random_state (42): Ensures reproducible results (same trees every run)
    
    ADVANTAGES:
    - Captures non-linear patterns and feature interactions
    - Robust to outliers and noisy data
    - Less likely to overfit than a single deep tree
    - Provides feature importances
    
    LIMITATIONS:
    - Slower to train and predict than linear regression
    - Less interpretable (can't explain predictions easily)
    - Requires more hyperparameter tuning
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training target (stock prices)
        n_estimators: Number of trees in the forest (default 100)
        max_depth: Maximum depth of each tree (default 10)
        random_state: Random seed for reproducibility (default 42)
        
    Returns:
        Trained Random Forest model
    """
    print("\n" + "="*80)
    print("TRAINING: Random Forest Regressor (Complex Model)")
    print("="*80)
    
    # Initialize Random Forest with chosen hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,      # Number of trees
        max_depth=max_depth,            # Maximum tree depth (prevents overfitting)
        random_state=random_state,      # For reproducibility
        n_jobs=-1,                      # Use all CPU cores (faster training)
        min_samples_split=5,            # Minimum samples to split a node (prevents overfitting)
        min_samples_leaf=2              # Minimum samples in leaf (prevents overfitting)
    )
    
    # Train the forest
    model.fit(X_train, y_train)
    
    # Log model details
    print(f"  ✓ Training complete!")
    print(f"    Number of trees: {n_estimators}")
    print(f"    Max depth per tree: {max_depth}")
    print(f"    Number of features: {X_train.shape[1]}")
    print(f"    Random state: {random_state} (reproducible results)")
    
    return model


def interpret_linear_regression(model: LinearRegression, feature_names: list) -> pd.DataFrame:
    """
    Interpret Linear Regression coefficients in plain English.
    
    WHAT ARE COEFFICIENTS?
    Each coefficient tells us the relationship between a feature and the target.
    
    HOW TO READ THEM:
    - Coefficient = +5.2 for "Volume_change":
      "A 1-unit increase in volume change leads to $5.20 increase in next day's price"
    
    - Coefficient = -3.1 for "Sentiment_change":
      "A 1-unit increase in sentiment change leads to $3.10 DECREASE in next day's price"
      (Negative = inverse relationship)
    
    IMPORTANT CONTEXT:
    - Remember features are SCALED (mean=0, std=1) before training
    - So "1-unit increase" means "1 standard deviation increase"
    - Larger absolute value = stronger influence on price
    
    ACTIONABLE INSIGHTS:
    - Positive coefficients: Buy when this feature increases
    - Negative coefficients: Sell when this feature increases
    - Near-zero coefficients: This feature doesn't help much
    
    Args:
        model: Trained Linear Regression model
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their coefficients sorted by importance
    """
    print("\n" + "="*80)
    print("LINEAR REGRESSION INTERPRETATION")
    print("="*80)
    
    # Create dataframe of coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Sort by absolute value (importance is about magnitude, not sign)
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance (sorted by magnitude):")
    print(coef_df[['Feature', 'Coefficient']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("HOW TO INTERPRET:")
    print("-"*80)
    print("✓ Positive coefficient = Feature increase → Price increase")
    print("✓ Negative coefficient = Feature increase → Price decrease")
    print("✓ Larger absolute value = Stronger influence")
    print("✓ Near-zero coefficient = Feature doesn't help prediction")
    print("-"*80)
    
    # Provide plain-English interpretation of top features
    if len(coef_df) > 0:
        top_feature = coef_df.iloc[0]
        direction = "increase" if top_feature['Coefficient'] > 0 else "decrease"
        print(f"\nMost Important Feature: {top_feature['Feature']}")
        print(f"  When this feature increases by 1 std dev, price tends to {direction} by ${abs(top_feature['Coefficient']):.2f}")
    
    return coef_df


def get_feature_importances(model: RandomForestRegressor, feature_names: list) -> pd.DataFrame:
    """
    Get feature importances from Random Forest model.
    
    WHAT ARE FEATURE IMPORTANCES?
    They measure how much each feature contributes to reducing prediction error
    across all trees in the forest.
    
    HOW THEY'RE CALCULATED:
    - When a tree splits on a feature, it reduces prediction error by some amount
    - Feature importance = average error reduction from that feature across all trees
    - Higher importance = feature is used more often and reduces error more
    
    HOW TO READ THEM:
    - Values sum to 1.0 (100%)
    - 0.45 = This feature accounts for 45% of the model's predictive power
    - 0.02 = This feature barely helps (only 2% contribution)
    
    DIFFERENCE FROM LINEAR REGRESSION COEFFICIENTS:
    - Coefficients: Show direction AND magnitude of relationship
    - Importances: Show ONLY how useful a feature is (no direction)
    
    ACTIONABLE INSIGHTS:
    - High importance: Monitor this feature closely, it drives predictions
    - Low importance: Could potentially remove this feature to simplify model
    - Surprising importances: May reveal unexpected market patterns
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and importances sorted by importance
    """
    print("\n" + "="*80)
    print("RANDOM FOREST FEATURE IMPORTANCES")
    print("="*80)
    
    # Extract importances from trained model
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Add percentage representation
    importance_df['Importance_Pct'] = (importance_df['Importance'] * 100).round(2)
    
    print("\nFeature Importance (sorted):")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.4f} ({row['Importance_Pct']:5.2f}%)")
    
    print("\n" + "-"*80)
    print("HOW TO INTERPRET:")
    print("-"*80)
    print("✓ Higher importance = More useful for prediction")
    print("✓ Values sum to 100%")
    print("✓ >20% = Very important feature")
    print("✓ <5% = Marginally helpful feature")
    print("-"*80)
    
    # Highlight the most important feature
    if len(importance_df) > 0:
        top_feature = importance_df.iloc[0]
        print(f"\nMost Important Feature: {top_feature['Feature']}")
        print(f"  Accounts for {top_feature['Importance_Pct']:.1f}% of the model's predictive power")
    
    return importance_df


def save_models(lr_model: LinearRegression, rf_model: RandomForestRegressor, 
                scaler: StandardScaler, output_dir: str = "models"):
    """
    Save trained models and scaler to disk.
    
    Args:
        lr_model: Trained Linear Regression model
        rf_model: Trained Random Forest model
        scaler: Fitted StandardScaler
        output_dir: Directory to save models
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving models to {output_dir}...")
    
    with open(output_path / 'linear_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open(output_path / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Models saved successfully")


def training_pipeline(X: pd.DataFrame, y: pd.Series, feature_names: list,
                     test_size: float = 0.2) -> dict:
    """
    Complete training pipeline.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary containing models, scaler, and data splits
    """
    # Split data (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train Linear Regression
    lr_model = train_linear_regression(X_train_scaled, y_train)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # Interpret models
    coef_df = interpret_linear_regression(lr_model, feature_names)
    importance_df = get_feature_importances(rf_model, feature_names)
    
    # Save models
    save_models(lr_model, rf_model, scaler)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'coef_df': coef_df,
        'importance_df': importance_df
    }


if __name__ == "__main__":
    # Test the training pipeline
    from preprocess import preprocess_pipeline
    from feature_engineering import feature_engineering_pipeline
    
    data_path = "../data/data.csv"
    stock_price_path = "../data/stock_price.csv"
    
    # Preprocess and engineer features
    df = preprocess_pipeline(data_path, stock_price_path)
    df_full, X, y, feature_names = feature_engineering_pipeline(df)
    
    # Train models
    results = training_pipeline(X, y, feature_names)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
