import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load your flattened and cleaned dataset
df = pd.read_csv("flattened_wildfire_data.csv")

# 2. Drop irrelevant or non-numeric columns
df = df.select_dtypes(include=["number"]).dropna()
df = df.drop('Std_confidence', axis=1)

# 3. Separate features and target
X = df.drop("Mean_confidence", axis=1).values
y = df["Mean_confidence"].values.reshape(-1, 1)

# 4. Normalize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 5. Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 6. Train-test split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 7. Define Neural Network
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = RegressionNN(input_dim=X.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Train model
epochs = 200
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 9. Evaluate
model.eval()
y_actual = []
y_pred = []
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        y_actual.extend(yb.numpy())
        y_pred.extend(pred.numpy())

# 10. Download weights
torch.save(model.state_dict(), "./models/weather_pred.pth")

# 11. Unscale and cap predictions
y_actual_unscaled = scaler_y.inverse_transform(y_actual)
y_pred_unscaled = scaler_y.inverse_transform(y_pred)
y_pred_capped = [[min(100, val[0])] for val in y_pred_unscaled]

# 12. Print metrics
mse = mean_squared_error(y_actual_unscaled, y_pred_capped)
mae = mean_absolute_error(y_actual_unscaled, y_pred_capped)
r2 = r2_score(y_actual_unscaled, y_pred_capped)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# 13. Plot
plt.figure(figsize=(8, 5))
plt.plot([75, 100], [75, 100], 'r--', label='Perfect Prediction')
plt.scatter(y_actual_unscaled, y_pred_capped, alpha=0.5)
plt.xlabel("Actual Confidence")
plt.ylabel("Predicted Confidence (Capped at 100)")
plt.title("Neural Network Predictions vs Actual")
plt.grid(True)
plt.legend()
plt.show()
