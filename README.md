# Stock Price Prediction - Machine Learning Project

## Overview

This project implements a machine learning pipeline to predict next-day stock prices using day-over-day changes in market data. I built this as a complete end-to-end learning exercise to understand how to approach a time-series regression problem.

**What I'm trying to predict:** Tomorrow's stock price, using what changed in the market today.

**Key insight:** Markets don't react to absolute values (e.g., "volume is 1 million") but to changes (e.g., "volume increased by 50,000"). This project builds on that understanding.

---

## The Problem & My Approach

When I started, the main challenges were:

1. **Time-series data isn't like other data** - You can't shuffle it and pretend the order doesn't matter
2. **Single-feature prediction is hard** - With only one market indicator, predicting stock prices is genuinely difficult
3. **Need to think about data alignment** - Today's changes should predict tomorrow's price (not the same day)

I learned that having a clear mental model of the problem before writing code was crucial. Before touching Python, I thought through: "What are features? What's the target? When would this model be used?"

---

## Project Structure

```
Stock Price Prediction/
├── data/
│   ├── data.csv                 # Market indicator (1 feature, 3902 rows)
│   └── stock_price.csv          # Stock prices (3840 rows)
│
├── src/
│   ├── preprocess.py            # Load, clean, align datasets
│   ├── feature_engineering.py   # Compute day-over-day changes
│   ├── train.py                 # Train Linear Regression & Random Forest
│   └── evaluate.py              # Evaluate models with metrics & plots
│
├── notebooks/
│   └── exploration.ipynb        # Interactive data exploration
│
├── models/                      # Saved trained models (generated)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── outputs/                     # Visualizations (generated)
│   ├── *_predictions.png
│   └── *_residuals.png
│
├── main.py                      # Run everything with one command
└── requirements.txt             # Python dependencies
```

---

## Getting Started

### Step 1: Set Up Your Environment

```bash
# Navigate to project
cd "Stock Price Prediction"

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

```bash
python main.py
```

This will:

- Load and clean data from both CSV files
- Compute day-over-day changes for all features
- Split data (80% training, 20% testing) while preserving temporal order
- Train two models: Linear Regression (simple) and Random Forest (complex)
- Evaluate both models and generate 6 visualization plots
- Save trained models to `models/` directory
- Save visualizations to `outputs/` directory

The entire process takes about 20-30 seconds.

---

## How It Works: The Technical Details

### 1. Data Preprocessing

**What I do:** Load two CSV files, check they're clean, and align them on dates.

**Why it matters:** The market data and stock prices came from different sources. They had different numbers of rows because trading didn't happen on every date.

**Key decisions:**

- **Inner join on dates:** Only use dates present in BOTH files. Better to have less data that's complete than partial data.
- **Forward-fill missing values:** If a value is missing, assume it stays the same until the next known value. This preserves time-series patterns better than mean imputation.
- **Sort chronologically:** CRITICAL for time-series. Never shuffle the data.

**Code example:**

```python
# Align datasets on Date column (inner join = only keep matching dates)
merged_df = pd.merge(data_df, stock_price_df, on='Date', how='inner')
```

### 2. Feature Engineering

**What I do:** Create features that models can actually learn from.

**The insight:** Stock prices react to CHANGES, not absolute values. If volume suddenly spikes, prices often move.

**Mathematical formula:**

```
ΔX(t) = X(t) - X(t-1)
```

For example, if volume on Monday = 1,000 and Tuesday = 1,200:

- Tuesday's feature = 1,200 - 1,000 = +200 (volume increased)

**Temporal alignment (crucial):**

```
Features at time t (changes from yesterday to today)
    ↓
Predict target at time t+1 (tomorrow's price)
```

This ensures we only use past information to predict the future—no "cheating" by using future data!

### 3. Model Training

I trained two models to compare approaches:

**Linear Regression:**

- Formula: `Price = β₀ + β₁×change₁ + β₂×change₂ + ...`
- Pros: Simple, fast, coefficients are interpretable
- Cons: Assumes linear relationships (real markets are more complex)
- My takeaway: Good for a baseline

**Random Forest:**

- Builds many decision trees, averages their predictions
- Pros: Captures non-linear patterns (e.g., "high volume AND negative sentiment")
- Cons: Slower, harder to explain why it makes predictions
- My takeaway: Often more accurate but less interpretable

**Why both models?**
If a complex model doesn't beat a simple one, something's probably wrong with my features or data. This forced me to think critically rather than just using the most sophisticated model.

**Key best practice: No shuffling**
Time-series data MUST maintain temporal order during train-test split.

- Training set: First 80% of dates (old data)
- Test set: Last 20% of dates (recent data)
- This mimics real-world scenario: Learn from history, predict the future

### 4. Model Evaluation

**Metrics I track:**

1. **MAE (Mean Absolute Error)** - Average prediction error in dollars

   - Easy to interpret: "On average, my predictions are off by $X"
   - Lower is better

2. **RMSE (Root Mean Squared Error)** - Like MAE but penalizes large errors more

   - If RMSE >> MAE, I have some really bad predictions
   - Lower is better

3. **R² Score** - What proportion of price variation does my model explain?
   - 1.0 = Perfect, 0.0 = Useless, negative = Worse than just guessing the average
   - Higher is better
   - **Critical insight:** Negative R² is a RED FLAG that something's very wrong

**What I learned from my results:**

- Both models got negative R² scores (-15)
- This means they're WORSE than just predicting the average price
- Why? Single-feature prediction is incredibly hard. Stock markets are complex.
- **This is honest and important to report:** It means I need better features, more data, or a different approach entirely.

---

## Results & Honest Assessment

After running with real market data (3,800 samples, 15 years):

| Metric | Linear Regression | Random Forest |
| ------ | ----------------- | ------------- |
| MAE    | $2,391            | $2,421        |
| RMSE   | $2,477            | $2,540        |
| R²     | -14.96            | -15.79        |

**What this means:**

- Linear Regression slightly outperforms Random Forest (1.3% better MAE)
- But BOTH models fail badly (negative R² means worse than average)
- The error ($2,400) is ~48% of the average stock price—really poor
- With only ONE feature (daily changes), we can't capture market dynamics well enough

**Why did the model struggle?**

1. **Single feature is insufficient** - Real markets depend on many factors (interest rates, news, sentiment, technical patterns)
2. **Limited historical data** - 3,800 samples over 15 years sounds like a lot but markets change
3. **Structural complexity** - Stock prices follow non-stationary patterns that are hard to predict

**What I would do next:**

- Add more features (moving averages, volatility, technical indicators)
- Try time-series models (ARIMA, LSTM) designed specifically for temporal data
- Incorporate external data (sentiment, economic indicators)
- Consider ensemble methods combining multiple models

**Important learning:** Sometimes models don't work well, and that's okay. The goal is to understand WHY, not just accept whatever numbers come out.

---

## Using the Models

### Making Predictions on New Data

```python
import pickle
import numpy as np
from src.preprocess import StandardScaler

# Load trained model and scaler
with open('models/linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# New day's feature (day-over-day change in market indicator)
new_change = np.array([[42.5]])  # Example: +42.5 change from yesterday

# Scale using same transformation as training
scaled = scaler.transform(new_change)

# Predict
prediction = model.predict(scaled)
print(f"Predicted tomorrow's price: ${prediction[0]:.2f}")
```

---

## File-by-File Explanation

### `src/preprocess.py`

Handles the "garbage in, garbage out" problem. Real data is messy. This module:

- Loads CSVs and validates they have required columns
- Checks for and handles missing values
- Removes duplicate dates
- Aligns datasets on dates (inner join = safe approach)

**Key function:** `preprocess_pipeline()` - orchestrates all steps

### `src/feature_engineering.py`

Transforms raw values into features models can learn from.

- Computes daily changes using pandas `.diff()`
- Creates target variable (next day's price) using `.shift(-1)`
- Validates that changes are reasonable (sanity checks)

**Key insight:** This is where I explicitly implement the mathematical formula for changes.

### `src/train.py`

Trains two different models and extracts insights:

- **Linear Regression:** Fits a line through the data, provides interpretable coefficients
- **Random Forest:** Builds ensemble of decision trees, provides feature importances
- **Key principle:** Feature scaling (StandardScaler) ensures all features are comparable

**Important practice:** Uses `random_state=42` for reproducibility. Same seed = same results every time.

### `src/evaluate.py`

Grades the models honestly:

- Calculates MAE, RMSE, R²
- **Sanity checks:** Are predictions negative? Are they in a reasonable range? Do they vary?
- Generates 6 plots showing predictions and errors
- Provides interpretation guidance (not just raw numbers)

**Philosophy:** Metrics without interpretation are meaningless. I want to explain what the numbers mean.

### `main.py`

Orchestrates everything in the right order:

1. Preprocess data
2. Engineer features
3. Train models
4. Evaluate performance

Running `python main.py` = Run the complete pipeline.

---

## Visualizations Generated

The pipeline creates 6 PNG plots in `outputs/`:

1. **Linear Regression - Predictions:** Actual vs predicted prices over time + scatter plot
2. **Random Forest - Predictions:** Same for Random Forest model
3. **Linear Regression - Residuals:** How much each prediction was off + distribution of errors
4. **Random Forest - Residuals:** Error analysis for Random Forest
5. **Feature Importances:** Bar chart showing which features matter most (Random Forest)
6. **Coefficients:** Bar chart showing relationship strength (Linear Regression)

**How to use visualizations:**

- **Predictions plot:** Do predictions follow actual prices? (Should mostly match if model works)
- **Residuals plot:** Are errors randomly scattered or do they show patterns? (Random is good, patterns suggest systematic errors)
- **Feature importance:** Which features drive predictions? (Helps understand what the model learned)
- **Coefficients:** Positive or negative relationships? (Tells us interpretable insights)

---

## Dependencies

- **pandas**: Data manipulation (loading, transforming, merging)
- **numpy**: Numerical operations (arrays, math)
- **scikit-learn**: Machine learning (models, metrics, scaling)
- **matplotlib** & **seaborn**: Visualization (creating plots)
- **jupyter**: Interactive notebooks for exploration

See `requirements.txt` for exact versions.

---

## What I Learned Building This

1. **Data preparation matters more than model choice**

   - I spent more time cleaning data and engineering features than training models
   - A simple model with good features beats a complex model with bad features

2. **Temporal order is sacred in time-series**

   - One mistake: shuffling time-series data = misleadingly good results
   - Right way: train on history, test on future (respects temporal order)

3. **Negative results are data too**

   - Getting R² = -15 is bad, but honestly reporting it matters
   - If a model fails, it tells you something about the problem
   - Single feature can't predict complex markets (important finding!)

4. **Interpretation > accuracy**

   - Numbers without context are useless
   - I included sanity checks, error analysis, and plain-English explanations
   - A model I understand (Linear Regression) is more useful than a black box

5. **Reproducibility requires discipline**
   - Fixed random seed → same results every run
   - Fit scaler on training data only → no data leakage
   - Version control code → can track what changed
   - These seem small but compound into reliability

---

## Challenges & How I Overcame Them

**Challenge 1: Column name mismatches**

- Datasets had different column names (one had 'Price', other 'Stock_Price')
- Solution: Dynamic column detection (find non-Date column and standardize names)
- Lesson: Make code flexible enough to handle real-world messiness

**Challenge 2: Data loss from merging**

- Inner join removed dates that didn't exist in both datasets
- Alternative was outer join but more missing values
- Solution: Made this transparent in logs so you know exactly how much data was lost

**Challenge 3: Negative R² results**

- Initial reaction: "Something's broken"
- Better understanding: "Model truly can't do this task well with current features"
- Lesson: Negative results are informative—don't just ignore them

**Challenge 4: Producing readable code**

- First version: Minimal comments, unclear variable names
- Better version: Detailed docstrings, math formulas in comments, clear logic flow
- Lesson: Code for humans first, computers second

---

## Contact & Questions

Built by: Nancy  
Date: January 2026  
Purpose: Machine Learning course/assignment

For questions about the code or approach, see the detailed comments throughout `src/` files.

---

## Summary

This project taught me that **good machine learning is 80% preparation and evaluation, 20% model selection**. The actual ML models are straightforward; the hard part is understanding the problem, preparing clean data, and honestly assessing results.

Most importantly: negative results teach you something. This model failed to predict prices with a single feature—that's not a flaw, it's a finding about the complexity of markets.

**Best takeaway:** The goal isn't always to build a perfect model. Sometimes the goal is to understand a problem well enough to know what would actually be needed to solve it.
