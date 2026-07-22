# Employee Resignation Classifier (Random Forest Classifier)

This project implements an end-to-end Machine Learning pipeline using a **Random Forest Classifier** to predict employee resignation (`Resigned`) from the `Extended_Employee_Performance_and_Productivity_Data.csv` dataset.

## Features & Tech Stack
- **Python 3**
- **Data Preprocessing & Encoding**: Drops noise columns (`Employee_ID`, `Hire_Date`), performs One-Hot Encoding for categorical features, and standardizes numerical variables.
- **Handling Class Imbalance**: The dataset is highly imbalanced (~10% resigned, 90% stayed). The pipeline uses stratified splits and `class_weight='balanced'` in the Random Forest to ensure robust training.
- **Scikit-Learn**: Model building, scaling, encoding, and metric computations.
- **Visual Analytics**: Heatmap of the Confusion Matrix, ROC-AUC Curve, Precision-Recall Curve, and Feature Importance distribution charts.

## Directory Structure
```text
├── Extended_Employee_Performance_and_Productivity_Data.csv  # Dataset (100k rows)
├── requirements.txt                                          # Python libraries needed
├── setup.bat                                                 # Windows setup helper script
├── main.py                                                   # Main orchestration runner
├── README.md                                                 # Project Documentation
├── src/                                                      # Codebase Source Module
│   ├── __init__.py
│   ├── data_preprocessing.py                                 # Loads, cleans, and encodes data
│   ├── model_training.py                                     # Configures and trains RF model
│   └── model_evaluation.py                                   # Evaluates and creates charts
└── outputs/                                                  # Saved models & visual plots
    ├── preprocessor.joblib                                   # Saved feature scaler/encoder
    ├── random_forest_model.joblib                            # Saved Random Forest model
    ├── classification_report.txt                             # Evaluation report text file
    ├── feature_importances.csv                               # Ranked feature importances data
    ├── feature_importances.png                               # Visual plot of top 15 features
    ├── confusion_matrix.png                                  # Heatmap of classifications
    ├── roc_curve.png                                         # Receiver Operating Characteristic plot
    └── pr_curve.png                                          # Precision-Recall plot
```

## Setup & Running Instructions

### Windows (Quick Start)
Double-click `setup.bat` in your file explorer, or run it from CMD/PowerShell:
```cmd
setup.bat
```
This script will:
1. Initialize a Python virtual environment (`venv`).
2. Upgrade `pip` and install all dependencies in `requirements.txt`.
3. Execute the ML pipeline `python main.py`.

### Manual Instructions (Cross-platform)
1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   source venv/Scripts/activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the project**:
   ```bash
   python main.py
   ```

## Model Evaluation Outputs
After execution, explore the `outputs/` folder for visual plots detailing model quality, confusion metrics, and feature relevance.
