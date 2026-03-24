# Customer Lifetime Value (CLV) Predictor: Team Assignments

## Project Status
- **Completed:** Environment Setup, Exploratory Data Analysis (EDA), Data Cleaning & Preprocessing *(Mayank)*
- **Pending:** Feature Engineering, Modeling, Deployment *(Harshita, Simran, Yash)*

---

## 1. Harshita: Feature Engineering & Target Definition
**Role:** Data & Feature Engineer

**Key Responsibilities:**
1. Start with the preprocessed dataset: `data/processed/customer_details_processed.csv`.
2. Create the `notebooks/03_feature_engineering.ipynb` file.
3. **Calculate RFM:** Extract Recency, Frequency, and Monetary scores from the dataset columns.
4. **Define LTV:** Create the mathematical target variable column combining baseline spending and frequency to define the "Customer Lifetime Value".
5. Save the final ML-ready dataset (e.g., `data/processed/final_ml_data.csv`).

---

## 2. Simran: Machine Learning Modeling & Evaluation
**Role:** Machine Learning Engineer

**Key Responsibilities:**
1. Take Harshita's finalized dataset and build `notebooks/04_model_training.ipynb`.
2. **Train/Test Split:** Perform a standard 80/20 train/test split.
3. **Baseline Models:** Train predictive algorithms like Linear Regression, Random Forest, and XGBoost Regressors.
4. **Evaluate & Tune:** Assess performance using RMSE, MAE, and R-Squared. Tune hyperparameters to maximize accuracy.
5. Export the winning model artifact (e.g., `best_model.pkl`) into the `models/` directory using `joblib` or `pickle`.

---

## 3. Yash: Streamlit Dashboard & Deployment
**Role:** MLOps & App Developer

**Key Responsibilities:**
1. Develop the `app.py` script using the **Streamlit** Python framework.
2. Load Simran's best saved model (`models/best_model.pkl`).
3. **Build the Interface:** Create an intuitive UI (sliders, dropdowns) for stakeholders to input a customer profile (e.g., Age, Subscription Status, Purchase Frequency).
4. **Real-time Predictions:** Feed those inputs directly into the ML model to predict and display the customer's expected Lifetime Value.
5. Display key EDA charts on the dashboard for extra context.
