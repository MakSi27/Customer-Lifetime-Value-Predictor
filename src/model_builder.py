from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# 1. Linear Regression

def get_linear_regression():
    return LinearRegression()


# 2. Random Forest

def get_random_forest():
    return RandomForestRegressor(random_state=42)


# 3. XGBoost

def get_xgboost():
    return XGBRegressor(random_state=42)