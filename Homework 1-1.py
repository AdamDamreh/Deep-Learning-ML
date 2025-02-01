import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

transform = transforms.Compose([
     transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
])

# Load training and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256, 128], output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        
        # Validation Accuracy
        val_acc = test_model(model, test_loader)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_acc:.2f}%")

    return train_losses, train_accuracies, val_accuracies
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
model = MLP()
train_losses, train_acc, val_acc = train_model(model, train_loader, test_loader, epochs=20)
torch.save(model.state_dict(), "mlp_cifar10.pth")
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=testset.classes))


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1,21), train_losses, label = "Training Loss", color = "blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,21), train_acc, label = "Training Accuracy", color = "green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Training Vs. Validation Accuracy Over Epochs")
plt.legend()

plt.show()

model_wide = MLP(hidden_sizes=[1024, 512, 256])
train_model(model_wide, train_loader, test_loader, epochs=20)
print("\nTraining Wider Model...")
train_losses_wide, train_acc_wide, val_acc_wide = train_model(model_wide, train_loader, test_loader, epochs=20)
torch.save(model_wide.state_dict(), "mlp_wide_cifar10.pth")
evaluate_model(model_wide, test_loader)

# Deeper Network
class DeepMLP(nn.Module):
   def __init__(self, input_size=3072, hidden_sizes=[1024, 512, 512, 256], output_size=10):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

   def forward(self, x):
       x = x.view(x.size(0), -1)
       x = self.relu(self.fc1(x))
       x = self.relu(self.fc2(x))
       x = self.relu(self.fc3(x))
       x = self.relu(self.fc4(x))
       x = self.fc5(x)
       return x

model_deep = DeepMLP()
print("\nTraining Deeper Model...")
train_losses_deep, train_acc_deep, val_acc_deep = train_model(model_deep, train_loader, test_loader, epochs=20)
torch.save(model_deep.state_dict(), "mlp_deep_cifar10.pth")
evaluate_model(model_deep, test_loader)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1,21), train_losses_wide, label="Wider Model", color="blue")
plt.plot(range(1,21), train_losses_deep, label="Deeper Model", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,21), train_acc_wide, label="Wider Model", color="blue")
plt.plot(range(1,21), train_acc_deep, label="Deeper Model", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Training Accuracy Over Epochs")
plt.legend()

plt.show()
