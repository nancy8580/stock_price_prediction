"""
Model Evaluation Module

This module evaluates model performance and provides actionable insights.

WHY EVALUATION MATTERS:
Training a model is only half the battle. We need to know:
1. How accurate is it? (metrics)
2. Where does it fail? (error analysis)
3. Can we trust it for trading decisions? (sanity checks)

EVALUATION STRATEGY:
1. Quantitative metrics (MAE, RMSE, R²) - objective performance measurement
2. Visual analysis (plots) - identify patterns in errors
3. Sanity checks - ensure predictions make sense (not negative prices, reasonable scale)
4. Model comparison - which model performs better?

THREE KEY METRICS EXPLAINED:

1. MAE (Mean Absolute Error):
   - Average of |actual - predicted|
   - Easy to interpret: "On average, predictions are off by $X"
   - Units: same as stock price (dollars)
   - Example: MAE=$50 means average error is $50
   
2. RMSE (Root Mean Squared Error):
   - Square root of average squared errors
   - Penalizes large errors more heavily than MAE
   - Units: same as stock price (dollars)
   - Always ≥ MAE (equality only if all errors are identical)
   - Example: RMSE=$75, MAE=$50 means some predictions are very wrong
   
3. R² (R-squared Coefficient of Determination):
   - Proportion of variance explained by the model
   - Range: -∞ to 1.0
   - Interpretation:
     * 1.0 = Perfect predictions
     * 0.0 = Model is no better than predicting the mean
     * Negative = Model is WORSE than just predicting the mean (bad sign!)
   - Example: R²=0.75 means model explains 75% of price variation

WHAT MAKES A \"GOOD\" MODEL?
- Stock prediction is HARD. Even professional models struggle.
- Realistic expectations:
  * MAE < 5% of average price: Excellent
  * MAE < 10% of average price: Good
  * MAE > 20% of average price: Poor
  * R² > 0.5: Good explanatory power
  * R² near 0 or negative: Model is struggling
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str = \"Model\") -> dict:
    """
    Calculate evaluation metrics with sanity checks.
    
    This function computes standard regression metrics and performs validation
    to ensure predictions are reasonable.
    
    SANITY CHECKS:
    1. Predictions should be in a reasonable range (not negative, not extreme)
    2. Predictions should vary (not all identical)
    3. Scale should match actual values
    
    Args:
        y_true: Actual stock prices
        y_pred: Predicted stock prices
        model_name: Name of model (for warning messages)
        
    Returns:
        Dictionary with metrics: MAE, RMSE, R², plus diagnostic info
    """
    # Sanity Check 1: Are there any negative predictions?
    # Stock prices can't be negative!
    if (y_pred < 0).any():
        num_negative = (y_pred < 0).sum()
        warnings.warn(f"⚠ {model_name}: {num_negative} negative predictions! Stock prices should be positive.", UserWarning)
        # Clip negative predictions to zero (conservative fix)
        y_pred = np.maximum(y_pred, 0)
    
    # Sanity Check 2: Are predictions in a reasonable range?
    actual_min, actual_max = y_true.min(), y_true.max()
    pred_min, pred_max = y_pred.min(), y_pred.max()
    
    if pred_max > actual_max * 2:
        warnings.warn(f"⚠ {model_name}: Maximum prediction (${pred_max:.2f}) is more than 2x the maximum actual price (${actual_max:.2f})", UserWarning)
    if pred_min < actual_min * 0.5:
        warnings.warn(f"⚠ {model_name}: Minimum prediction (${pred_min:.2f}) is less than 50% of minimum actual price (${actual_min:.2f})", UserWarning)
    
    # Sanity Check 3: Are predictions varying or all the same?
    pred_std = y_pred.std()
    if pred_std < 1.0:  # Less than $1 standard deviation
        warnings.warn(f"⚠ {model_name}: Predictions barely vary (std=${pred_std:.2f}). Model may not be learning.", UserWarning)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional diagnostic metrics
    avg_price = y_true.mean()
    mae_percentage = (mae / avg_price) * 100
    
    # Interpretation flags
    mae_quality = "Excellent" if mae_percentage < 5 else "Good" if mae_percentage < 10 else "Fair" if mae_percentage < 20 else "Poor"
    r2_quality = "Excellent" if r2 > 0.7 else "Good" if r2 > 0.5 else "Fair" if r2 > 0.3 else "Poor" if r2 > 0 else "Very Poor"
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAE_Percentage': mae_percentage,
        'MAE_Quality': mae_quality,
        'R2_Quality': r2_quality,
        'Avg_Price': avg_price,
        'Pred_Min': pred_min,
        'Pred_Max': pred_max
    }


def print_metrics(metrics: dict, model_name: str):
    """
    Print evaluation metrics with interpretation and context.
    
    Args:
        metrics: Dictionary with MAE, RMSE, R², and diagnostic info
        model_name: Name of the model
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - EVALUATION RESULTS")
    print(f"{'='*80}")
    
    # Print core metrics
    print(f"\nQuantitative Metrics:")
    print(f"  MAE (Mean Absolute Error):      ${metrics['MAE']:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${metrics['RMSE']:,.2f}")
    print(f"  R² Score (Coefficient of Determination): {metrics['R2']:.4f}")
    
    # Print context
    print(f"\nContext:")
    print(f"  Average actual price: ${metrics['Avg_Price']:,.2f}")
    print(f"  MAE as % of avg price: {metrics['MAE_Percentage']:.2f}%")
    print(f"  Prediction range: ${metrics['Pred_Min']:,.2f} to ${metrics['Pred_Max']:,.2f}")
    
    # Print interpretation
    print(f"\nInterpretation:")
    print(f"  ✓ Error magnitude: {metrics['MAE_Quality']}")
    print(f"    (On average, predictions are off by ${metrics['MAE']:,.2f})")
    print(f"  ✓ Explanatory power: {metrics['R2_Quality']}")
    if metrics['R2'] < 0:
        print(f"    ⚠ NEGATIVE R²! Model is worse than just predicting the average price.")
        print(f"      This suggests the features may not be predictive or the model isn't learning properly.")
    elif metrics['R2'] < 0.3:
        print(f"    ⚠ Low R². Model explains only {metrics['R2']*100:.1f}% of price variation.")
        print(f"      Consider: more features, different model architecture, or longer data collection.")
    else:
        print(f"    Model explains {metrics['R2']*100:.1f}% of price variation.")
    
    print(f"{'='*80}")


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, 
                     model_name: str, output_dir: str = "outputs"):
    """
    Plot actual vs predicted stock prices.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Time series comparison
    axes[0].plot(range(len(y_true)), y_true.values, label='Actual', marker='o', markersize=4, alpha=0.7)
    axes[0].plot(range(len(y_pred)), y_pred, label='Predicted', marker='x', markersize=4, alpha=0.7)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Stock Price ($)')
    axes[0].set_title(f'{model_name}: Actual vs Predicted Stock Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Stock Price ($)')
    axes[1].set_ylabel('Predicted Stock Price ($)')
    axes[1].set_title(f'{model_name}: Actual vs Predicted Scatter')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_predictions.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path / filename}")
    
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, 
                   model_name: str, output_dir: str = "outputs"):
    """
    Plot residual analysis to identify patterns in prediction errors.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    residuals = y_true.values - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Residuals over time
    axes[0].scatter(range(len(residuals)), residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Residual ($)')
    axes[0].set_title(f'{model_name}: Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name}: Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_residuals.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path / filename}")
    
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, 
                            output_dir: str = "outputs"):
    """
    Plot feature importances from Random Forest model.
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot top 10 features
    top_features = importance_df.head(10)
    
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Random Forest: Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    filename = "feature_importances.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path / filename}")
    
    plt.close()


def plot_coefficients(coef_df: pd.DataFrame, 
                      output_dir: str = "outputs"):
    """
    Plot Linear Regression coefficients.
    
    Args:
        coef_df: DataFrame with Feature and Coefficient columns
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot top 10 features by absolute coefficient
    top_features = coef_df.head(10)
    
    colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
    
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Linear Regression: Top 10 Coefficients by Magnitude')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    filename = "linear_regression_coefficients.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path / filename}")
    
    plt.close()


def compare_models(lr_metrics: dict, rf_metrics: dict):
    """
    Compare performance of both models with actionable insights.
    
    Args:
        lr_metrics: Metrics for Linear Regression
        rf_metrics: Metrics for Random Forest
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame({
        'Metric': ['MAE ($)', 'RMSE ($)', 'R²', 'MAE Quality', 'R² Quality'],
        'Linear Regression': [
            f"${lr_metrics['MAE']:,.2f}",
            f"${lr_metrics['RMSE']:,.2f}",
            f"{lr_metrics['R2']:.4f}",
            lr_metrics['MAE_Quality'],
            lr_metrics['R2_Quality']
        ],
        'Random Forest': [
            f"${rf_metrics['MAE']:,.2f}",
            f"${rf_metrics['RMSE']:,.2f}",
            f"{rf_metrics['R2']:.4f}",
            rf_metrics['MAE_Quality'],
            rf_metrics['R2_Quality']
        ]
    })
    
    print(\"\\n\" + comparison_df.to_string(index=False))
    
    # Determine winner for each metric (lower is better for MAE/RMSE, higher for R²)
    mae_winner = \"Random Forest\" if rf_metrics['MAE'] < lr_metrics['MAE'] else \"Linear Regression\"\n    rmse_winner = \"Random Forest\" if rf_metrics['RMSE'] < lr_metrics['RMSE'] else \"Linear Regression\"\n    r2_winner = \"Random Forest\" if rf_metrics['R2'] > lr_metrics['R2'] else \"Linear Regression\"\n    \n    # Calculate improvement percentages\n    mae_improvement = abs(rf_metrics['MAE'] - lr_metrics['MAE']) / max(rf_metrics['MAE'], lr_metrics['MAE']) * 100\n    r2_improvement = abs(rf_metrics['R2'] - lr_metrics['R2'])\n    \n    print(f\"\\nWinner by Metric:\")\n    print(f\"  \u2713 MAE:  {mae_winner} ({mae_improvement:.1f}% better)\")\n    print(f\"  \u2713 RMSE: {rmse_winner}\")\n    print(f\"  \u2713 R²:   {r2_winner} ({r2_improvement:.3f} points better)\")\n    \n    # Provide recommendation\n    print(f\"\\n{'='*80}\")\n    print(\"RECOMMENDATION:\")\n    print(f\"{'='*80}\")\n    \n    if mae_winner == rmse_winner == r2_winner:\n        print(f\"  \u2713 Clear winner: {mae_winner}\")\n        print(f\"    This model consistently outperforms across all metrics.\")\n    else:\n        print(f\"  \u26a0 Mixed results - different models win different metrics.\")\n        print(f\"    Consider: MAE for average case performance, R² for explanatory power.\")\n    \n    # Honest assessment of both models\n    if lr_metrics['R2'] < 0 and rf_metrics['R2'] < 0:\n        print(f\"\\n  \u26a0 CRITICAL: Both models have negative R² scores!\")\n        print(f\"    This means BOTH models are worse than just predicting the average price.\")\n        print(f\"    Possible causes:\")\n        print(f\"      - Features lack predictive power (need better/more features)\")\n        print(f\"      - Data quality issues (outliers, errors, insufficient samples)\")\n        print(f\"      - Market patterns are too complex for current features\")\n        print(f\"    Next steps: Feature engineering, collect more data, or try different approach.\")\n    elif max(lr_metrics['R2'], rf_metrics['R2']) < 0.3:\n        print(f\"\\n  \u26a0 Both models show weak predictive power (R² < 0.3).\")\n        print(f\"    Current features explain less than 30% of price variation.\")\n        print(f\"    Recommendations:\")\n        print(f\"      - Add more features (technical indicators, market data)\")\n        print(f\"      - Increase data collection period\")\n        print(f\"      - Consider time series models (ARIMA, LSTM)\")\n    \n    print(f\"{'='*80}\")


def evaluation_pipeline(results: dict):
    """
    Complete evaluation pipeline for both models.
    
    This orchestrates the entire evaluation process:
    1. Generate predictions from both models
    2. Calculate metrics with sanity checks
    3. Print results with interpretation
    4. Compare models with recommendations
    5. Generate visualizations for analysis
    
    Args:
        results: Dictionary from training pipeline containing models and data
    """
    print("\n" + "="*80)
    print("STARTING EVALUATION PIPELINE")
    print("="*80)
    
    # Get models and data
    lr_model = results['lr_model']
    rf_model = results['rf_model']
    X_test = results['X_test']
    y_test = results['y_test']
    coef_df = results['coef_df']
    importance_df = results['importance_df']
    
    # Make predictions
    print("\nGenerating predictions on test set...")
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    print(f"  \u2713 Linear Regression: {len(lr_pred)} predictions\")")
    print(f\"  \u2713 Random Forest: {len(rf_pred)} predictions\")")
    
    # Calculate metrics with sanity checks
    print("\nCalculating metrics with validation checks...")
    lr_metrics = calculate_metrics(y_test, lr_pred, \"Linear Regression\")
    rf_metrics = calculate_metrics(y_test, rf_pred, \"Random Forest\")
    
    # Print metrics with interpretation
    print_metrics(lr_metrics, \"Linear Regression\")
    print_metrics(rf_metrics, \"Random Forest\")
    
    # Compare models
    compare_models(lr_metrics, rf_metrics)
    
    # Generate plots
    print(f\"\\nGenerating visualizations ({6} plots)...\")
    plot_predictions(y_test, lr_pred, \"Linear Regression\")
    plot_predictions(y_test, rf_pred, \"Random Forest\")
    plot_residuals(y_test, lr_pred, \"Linear Regression\")
    plot_residuals(y_test, rf_pred, \"Random Forest\")
    plot_feature_importance(importance_df)
    plot_coefficients(coef_df)
    
    print(f\"\\n{'='*80}\")
    print(\"EVALUATION COMPLETE! Review outputs/ directory for visualizations.\")
    print(f\"{'='*80}\")


if __name__ == "__main__":
    # Test the evaluation pipeline
    from preprocess import preprocess_pipeline
    from feature_engineering import feature_engineering_pipeline
    from train import training_pipeline
    
    data_path = "../data/data.csv"
    stock_price_path = "../data/stock_price.csv"
    
    # Full pipeline
    df = preprocess_pipeline(data_path, stock_price_path)
    df_full, X, y, feature_names = feature_engineering_pipeline(df)
    results = training_pipeline(X, y, feature_names)
    
    # Evaluate
    evaluation_pipeline(results)
