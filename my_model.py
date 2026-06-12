import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#====================
# Data loading and preprocessing
#====================

X, y = worldcup_loader(n_samples=1000, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)

#====================
#Defining the architecture of the model
#====================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Define activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the flow of data
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)     # Add it to the flow.
        x = self.relu(x)    # Add an activation.
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = MLP(
    input_size=8,              # 8 features.
    hidden_size=10,            # 10 nodes in the layer.
    output_size=3).to(device)  # Make a final classification.
print(model)

#====================
# Training loop
#====================


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for epoch in range(epochs):
    # 1. Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 2. Backward pass and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()      # Compute gradients
    optimizer.step()     # Update weights

    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#====================
# Evaluation
#====================

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

from sklearn.metrics import classification_report

if device.type == 'cuda':
    y_pred = y_pred.cpu()

print(classification_report(y_test, y_pred.round()))
#====================