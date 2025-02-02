import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv("Housing.csv")

binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for feature in binary_features:
    data[feature] = data[feature].map({'yes': 1, 'no': 0})

data = pd.get_dummies(data, columns=['furnishingstatus'])

data = data.dropna(subset=['price'])  
data['area'] = data['area'].fillna(data['area'].median())
data['parking'] = data['parking'].fillna(data['parking'].median())

X = data.drop('price', axis=1)
y = data['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_val = scaler_y.transform(y_val.to_numpy().reshape(-1, 1))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64)         
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)         
        self.fc5 = nn.Linear(16, 1)          

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))  
        x = self.fc5(x)
        return x

input_dim = X_train.shape[1]  
model = MLP(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = criterion(y_val_pred, y_val)
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs. Validation Loss (Increased Complexity)")
plt.show()

from sklearn.metrics import r2_score

y_train_pred = model(X_train).detach().numpy()
y_val_pred = model(X_val).detach().numpy()

y_train_actual = scaler_y.inverse_transform(y_train.numpy())
y_val_actual = scaler_y.inverse_transform(y_val.numpy())
y_train_pred_actual = scaler_y.inverse_transform(y_train_pred)
y_val_pred_actual = scaler_y.inverse_transform(y_val_pred)

r2_train = r2_score(y_train_actual, y_train_pred_actual)
r2_val = r2_score(y_val_actual, y_val_pred_actual)

total_params = sum(p.numel() for p in model.parameters())

print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")
print(f"R^2 Train Score: {r2_train:.4f}")
print(f"R^2 Validation Score: {r2_val:.4f}")
print(f"Total Model Parameters: {total_params} (Increased from previous)")
