"""
Submission Verification Script

This script verifies that all required files and outputs are present
and the project is ready for submission.
"""

import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    exists = Path(dirpath).exists() and Path(dirpath).is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    print("=" * 80)
    print("SUBMISSION VERIFICATION CHECK")
    print("=" * 80)
    
    all_checks = []
    
    # Check code files
    print("\n📝 Code Files:")
    all_checks.append(check_file_exists("main.py", "Main execution script"))
    all_checks.append(check_file_exists("src/__init__.py", "Package init"))
    all_checks.append(check_file_exists("src/preprocess.py", "Preprocessing module"))
    all_checks.append(check_file_exists("src/feature_engineering.py", "Feature engineering"))
    all_checks.append(check_file_exists("src/train.py", "Training module"))
    all_checks.append(check_file_exists("src/evaluate.py", "Evaluation module"))
    
    # Check data files
    print("\n�� Data Files:")
    all_checks.append(check_file_exists("data/data.csv", "Input features"))
    all_checks.append(check_file_exists("data/stock_price.csv", "Stock prices"))
    
    # Check model files
    print("\n🤖 Model Files:")
    all_checks.append(check_file_exists("models/linear_regression.pkl", "Linear Regression model"))
    all_checks.append(check_file_exists("models/random_forest.pkl", "Random Forest model"))
    all_checks.append(check_file_exists("models/scaler.pkl", "StandardScaler"))
    
    # Check output files
    print("\n📈 Visualization Files:")
    all_checks.append(check_file_exists("outputs/linear_regression_predictions.png", "LR predictions"))
    all_checks.append(check_file_exists("outputs/random_forest_predictions.png", "RF predictions"))
    all_checks.append(check_file_exists("outputs/linear_regression_residuals.png", "LR residuals"))
    all_checks.append(check_file_exists("outputs/random_forest_residuals.png", "RF residuals"))
    all_checks.append(check_file_exists("outputs/feature_importances.png", "Feature importances"))
    all_checks.append(check_file_exists("outputs/linear_regression_coefficients.png", "Coefficients"))
    
    # Check documentation
    print("\n📚 Documentation Files:")
    all_checks.append(check_file_exists("README.md", "Main README"))
    all_checks.append(check_file_exists("QUICKSTART.md", "Quick start guide"))
    all_checks.append(check_file_exists("ASSESSMENT_SUBMISSION.md", "Assessment report ⭐"))
    all_checks.append(check_file_exists("EXECUTIVE_SUMMARY.md", "Executive summary"))
    all_checks.append(check_file_exists("SUBMISSION_CHECKLIST.md", "Submission checklist"))
    all_checks.append(check_file_exists("requirements.txt", "Dependencies"))
    all_checks.append(check_file_exists("LICENSE", "License file"))
    all_checks.append(check_file_exists(".gitignore", "Git ignore"))
    
    # Check notebook
    print("\n📓 Notebook Files:")
    all_checks.append(check_file_exists("notebooks/exploration.ipynb", "Exploratory notebook"))
    
    # Summary
    print("\n" + "=" * 80)
    total = len(all_checks)
    passed = sum(all_checks)
    print(f"VERIFICATION SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ ALL CHECKS PASSED - READY FOR SUBMISSION!")
    else:
        print(f"⚠️  {total - passed} files missing - please review")
    
    print("=" * 80)
    
    # File sizes
    print("\n📦 Project Statistics:")
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', '.'], capture_output=True, text=True)
        print(f"Total size: {result.stdout.split()[0]}")
    except:
        pass
    
    print(f"\n🎯 Main submission document: ASSESSMENT_SUBMISSION.md")
    print(f"📄 Quick reference: EXECUTIVE_SUMMARY.md")
    print(f"✅ Submission status: {'READY' if passed == total else 'INCOMPLETE'}")

if __name__ == "__main__":
    main()
