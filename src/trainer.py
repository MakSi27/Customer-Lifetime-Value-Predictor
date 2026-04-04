from sklearn.model_selection import train_test_split, GridSearchCV
from src.model_builder import (
    get_linear_regression,
    get_random_forest,
    get_xgboost
)

# 1. Data Splitting

def split_data(df):
   
    X = df.drop([
        "ltv",
        "customer_segment"
    ], axis=1)

    y = df["customer_segment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# 2. Linear Regression

def train_linear_regression(X_train, y_train):
    model = get_linear_regression()
    model.fit(X_train, y_train)
    return model


# 3. Random Forest 

def train_random_forest(X_train, y_train):

    rf = get_random_forest()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_


# 4. XGBoost 

def train_xgboost(X_train, y_train):

    xgb = get_xgboost()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_