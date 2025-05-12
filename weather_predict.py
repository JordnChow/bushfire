import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the CSV
df = pd.read_csv('./flattened_wildfire_data.csv')

X = df.drop(columns=['Mean_confidence'])
y = df['Mean_confidence']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load('fire_confidence_model.pkl')
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Predicted: {y_pred[0]:.2f}, Actual: {y_test.iloc[0]:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"Mean absolute error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
# Scatter plot of Actual (green) and Predicted (red)
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test.iloc[0:1000])), y_test.iloc[0:1000], label='Actual', color='green', marker='o')
plt.scatter(range(len(y_pred[0:1000])), y_pred[0:1000], label='Predicted', color='red', marker='x')
plt.title('Prediction vs Actual Mean_confidence')
plt.xlabel('Sample Index')
plt.ylabel('Mean_confidence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



