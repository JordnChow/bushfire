import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


# Load your data here
df = pd.read_csv("./flattened_wildfire_data.csv")  # Replace with actual file path
df = df.dropna()

# Separate features and target
X = df.drop(columns=['Mean_confidence'])
y = df['Mean_confidence']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and parameter grids
models = {
    'RandomForest': (RandomForestRegressor(), {
        'n_estimators': [100, 200],
        'max_depth': [x for x in range(1, 11)]
    }),
    'GradientBoosting': (GradientBoostingRegressor(), {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    'SVR': (Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ]), {
        'svr__C': [1, 10],
        'svr__kernel': ['rbf', 'linear']
    }),
    'Ridge': (Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ]), {
        'ridge__alpha': [0.1, 1.0, 10.0]
    })
}
# Run grid search for each model
results = []
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append((name, r2, mae, rmse, grid.best_params_))

# Display results
print("\nModel Comparison:")
for name, r2, mae, rmse, params in sorted(results, key=lambda x: -x[1]):
    print(f"{name}: R² = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, Best Params: {params}")
