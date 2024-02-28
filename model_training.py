import pandas as pd
import xgboost as xgb
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from joblib import dump

# Load the data
df = pd.read_csv('dataset/optimal_dataset.csv')

def round_to_next_odd(n):
    rounded = np.ceil(n)
    return np.where(rounded % 2 == 1, rounded, rounded + 1)


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# For the first model
X1_train = train_df[['p_dep', 'p_gate', 'p_res', 'p_read', 'l']]
y1_train = train_df['d']

X1_test = test_df[['p_dep', 'p_gate', 'p_res', 'p_read', 'l']]
y1_test = test_df['d']

# For the second model
X2_train = train_df[['d', 'l']]
y2_train = train_df['r']

X2_test = test_df[['d', 'l']]
y2_test = test_df['r']

# Hyperparameter tuning for first model
param_grid_xgb = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [50, 100, 150, 200]
}

xgb1 = xgb.XGBRegressor(objective ='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb1, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X1_train, y1_train)

# Predicting 'd' using the best estimator
best_xgb1 = grid_search.best_estimator_
y1_pred = best_xgb1.predict(X1_test)
y1_pred_new = round_to_next_odd(y1_pred)
print(best_xgb1)


# Hyperparameter tuning for second model
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Hyperparameter tuning for second model
rf1 = RandomForestRegressor()
grid_search2 = GridSearchCV(estimator=rf1, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search2.fit(X2_train, y2_train)

# Predicting 'r' using the best estimator
best_rf1 = grid_search2.best_estimator_
y2_pred = best_rf1.predict(X2_test)
y2_pred_new = np.ceil(y2_pred)
print(best_rf1)

# Save the first model
dump(grid_search.best_estimator_, 'trained_models/model_xgb.joblib')

# Save the second model
dump(grid_search2.best_estimator_, 'trained_models/model_rf.joblib')


# Metrics for 'd'
r2_d = r2_score(y1_test, y1_pred)
pearson_coef_d, _ = pearsonr(y1_test, y1_pred)
pearson_coef_d_rounded, _ = pearsonr(y1_test, y1_pred_new)
# print(f"R2 for 'd': {r2_d}")
print(f"Pearson Coefficient for 'd': {pearson_coef_d}")
print(f"Pearson Coefficient for 'd' rounded: {pearson_coef_d_rounded}")

# Metrics for 'r'
r2_r = r2_score(y2_test, y2_pred)
pearson_coef_r, _ = pearsonr(y2_test, y2_pred)
pearson_coef_r_rounded, _ = pearsonr(y2_test, y2_pred_new)
# print(f"R2 for 'r': {r2_r}")
print(f"Pearson Coefficient for 'r': {pearson_coef_r}")
print(f"Pearson Coefficient for 'r' rounded: {pearson_coef_r_rounded}")