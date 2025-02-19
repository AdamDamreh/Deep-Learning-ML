import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

class AlexNet_NoDropout(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_NoDropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_Dropout(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"[{model_name}] Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    torch.save(model.state_dict(), f"{model_name}.pth")


def evaluate_model(model, test_loader, model_name):
    model.to(device)
    model.eval()
    
    test_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    print(f"[{model_name}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_cifar10 = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_cifar10 = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_cifar100 = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
test_cifar100 = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)

train_loader_10 = DataLoader(train_cifar10, batch_size=64, shuffle=True)
test_loader_10 = DataLoader(test_cifar10, batch_size=64, shuffle=False)

train_loader_100 = DataLoader(train_cifar100, batch_size=64, shuffle=True)
test_loader_100 = DataLoader(test_cifar100, batch_size=64, shuffle=False)

model_cifar10_no_dropout = AlexNet_NoDropout(num_classes=10)
model_cifar10_dropout = AlexNet_Dropout(num_classes=10)
model_cifar100_no_dropout = AlexNet_NoDropout(num_classes=100)
model_cifar100_dropout = AlexNet_Dropout(num_classes=100)

criterion = nn.CrossEntropyLoss()
optimizer_10_no_dropout = optim.Adam(model_cifar10_no_dropout.parameters(), lr=0.001)
optimizer_10_dropout = optim.Adam(model_cifar10_dropout.parameters(), lr=0.001)
optimizer_100_no_dropout = optim.Adam(model_cifar100_no_dropout.parameters(), lr=0.001)
optimizer_100_dropout = optim.Adam(model_cifar100_dropout.parameters(), lr=0.001)

train_model(model_cifar10_no_dropout, train_loader_10, test_loader_10, criterion, optimizer_10_no_dropout, epochs=50, model_name="cifar10_no_dropout")
evaluate_model(model_cifar10_no_dropout, test_loader_10, model_name="cifar10_no_dropout")

train_model(model_cifar10_dropout, train_loader_10, test_loader_10, criterion, optimizer_10_dropout, epochs=50, model_name="cifar10_dropout")
evaluate_model(model_cifar10_dropout, test_loader_10, model_name="cifar10_dropout")

train_model(model_cifar100_no_dropout, train_loader_100, test_loader_100, criterion, optimizer_100_no_dropout, epochs=50, model_name="cifar100_no_dropout")
evaluate_model(model_cifar100_no_dropout, test_loader_100, model_name="cifar100_no_dropout")

train_model(model_cifar100_dropout, train_loader_100, test_loader_100, criterion, optimizer_100_dropout, epochs=50, model_name="cifar100_dropout")
evaluate_model(model_cifar100_dropout, test_loader_100, model_name="cifar100_dropout")
