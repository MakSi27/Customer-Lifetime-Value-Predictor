import pandas as pd
from src.data_loader import load_data
from src.processor import process_pipeline
from src.trainer import split_data, train_linear_regression, train_random_forest, train_xgboost
from src.evaluator import evaluate_model
import warnings

warnings.filterwarnings('ignore') # Suppress sklearn future warnings for clean output

def main():
    DATA_PATH = "data/processed/customer_details_processed.csv"
    
    print("--- 1. Loading Data ---")
    try:
        df = load_data(DATA_PATH)
        print(f"Data loaded successfully. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}.")
        return

    print("\n--- 2. Processing Pipeline ---")
    df = process_pipeline(df)
    print(f"LTV target created successfully! Final Data Shape: {df.shape}")
    print(f"Columns ready for training: {len(df.columns)}")

    print("\n--- 3. Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set has {X_train.shape[0]} rows.")
    print(f"Testing set has {X_test.shape[0]} rows.")

    print("\n--- 4. Training Models & Evaluation ---")
    
    # 4a. Linear Regression
    print("\n=> Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_results = evaluate_model(y_test, lr_preds)
    print(f"Linear Regression Results: {lr_results}")

    # 4b. Random Forest 
    print("\n=> Training Random Forest (GridSearch CV)...")
    print("   (This might take a minute...)")
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_results = evaluate_model(y_test, rf_preds)
    print(f"Random Forest Best Params: {rf_model.get_params()}")
    print(f"Random Forest Results: {rf_results}")

    # 4c. XGBoost
    print("\n=> Training XGBoost (GridSearch CV)...")
    print("   (This might take a minute...)")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_results = evaluate_model(y_test, xgb_preds)
    print(f"XGBoost Best Params: {xgb_model.get_params()}")
    print(f"XGBoost Results: {xgb_results}")

if __name__ == "__main__":
    main()
# done