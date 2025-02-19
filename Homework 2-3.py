import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=10, dropout_prob=0.0):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        layers = []
        layers.append(block(self.in_planes, planes, stride, dropout_prob))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet11(nn.Module):
    def __init__(self, block, layers, num_classes=10, dropout_prob=0.0):
        super(ResNet11, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        layers = []
        layers.append(block(self.in_planes, planes, stride, dropout_prob))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_dataloader(dataset, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == "CIFAR-10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset == "CIFAR-100":
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, num_classes

def train_and_evaluate(model, trainloader, testloader, num_epochs=50, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, val_loss, val_accuracy, train_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy_epoch = 100 * correct_train / total_train
        train_loss.append(running_loss / len(trainloader))
        val_loss_epoch, val_accuracy_epoch = evaluate(model, testloader, criterion)
        val_loss.append(val_loss_epoch)
        val_accuracy.append(val_accuracy_epoch)
        train_accuracy.append(train_accuracy_epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(trainloader):.4f}, Train Accuracy: {train_accuracy_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy_epoch:.4f}")

    return train_loss, val_loss, val_accuracy, train_accuracy

def evaluate(model, testloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(testloader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

def plot_results(train_loss, val_loss, val_accuracy, train_accuracy, model_name, dataset):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{model_name} Loss - {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy - {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

dataset_names = ['CIFAR-10', 'CIFAR-100']
dropout_options = [0.0, 0.5]  
models = [ResNet18, ResNet11]

trainloader_cifar10, testloader_cifar10, num_classes_cifar10 = get_dataloader("CIFAR-10")
print("ResNet11 CIFAR-10")
model_resnet11_0_cifar10 = ResNet11(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.0, num_classes=num_classes_cifar10)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet11_0_cifar10, trainloader_cifar10, testloader_cifar10, num_epochs=20)
torch.save(model_resnet11_0_cifar10.state_dict(), 'ResNet11_0_CIFAR10.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet11_0", "CIFAR-10")

print("ResNet11 Dropout CIFAR-10")
model_resnet11_5_cifar10 = ResNet11(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.5, num_classes=num_classes_cifar10)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet11_5_cifar10, trainloader_cifar10, testloader_cifar10, num_epochs=10)
torch.save(model_resnet11_5_cifar10.state_dict(), 'ResNet11_5_CIFAR10.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet11_5", "CIFAR-10")

print("ResNet18 CIFAR-10")
model_resnet18_0_cifar10 = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.0, num_classes=num_classes_cifar10)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet18_0_cifar10, trainloader_cifar10, testloader_cifar10, num_epochs=20)
torch.save(model_resnet18_0_cifar10.state_dict(), 'ResNet18_0_CIFAR10.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet18_0", "CIFAR-10")

print("ResNet18 Dropout CIFAR-10")
model_resnet18_5_cifar10 = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.5, num_classes=num_classes_cifar10)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet18_5_cifar10, trainloader_cifar10, testloader_cifar10, num_epochs=20)
torch.save(model_resnet18_5_cifar10.state_dict(), 'ResNet18_5_CIFAR10.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet18_5", "CIFAR-10")

trainloader_cifar100, testloader_cifar100, num_classes_cifar100 = get_dataloader("CIFAR-100")

print("ResNet11 CIFAR-100")
model_resnet11_0_cifar100 = ResNet11(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.0, num_classes=num_classes_cifar100)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet11_0_cifar100, trainloader_cifar100, testloader_cifar100, num_epochs=50)
torch.save(model_resnet11_0_cifar100.state_dict(), 'ResNet11_0_CIFAR100.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet11_0", "CIFAR-100")

print("ResNet11 Dropout CIFAR-100")
model_resnet11_5_cifar100 = ResNet11(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.5, num_classes=num_classes_cifar100)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet11_5_cifar100, trainloader_cifar100, testloader_cifar100, num_epochs=50)
torch.save(model_resnet11_5_cifar100.state_dict(), 'ResNet11_5_CIFAR100.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet11_5", "CIFAR-100")

print("ResNet18 CIFAR-100")
model_resnet18_0_cifar100 = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.0, num_classes=num_classes_cifar100)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet18_0_cifar100, trainloader_cifar100, testloader_cifar100, num_epochs=50)
torch.save(model_resnet18_0_cifar100.state_dict(), 'ResNet18_0_CIFAR100.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet18_0", "CIFAR-100")

print("ResNet18 Dropout CIFAR-100")
model_resnet18_5_cifar100 = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], dropout_prob=0.5, num_classes=num_classes_cifar100)
train_loss, val_loss, val_accuracy, train_accuracy = train_and_evaluate(model_resnet18_5_cifar100, trainloader_cifar100, testloader_cifar100, num_epochs=50)
torch.save(model_resnet18_5_cifar100.state_dict(), 'ResNet18_5_CIFAR100.pth')
plot_results(train_loss, val_loss, val_accuracy, train_accuracy, "ResNet18_5", "CIFAR-100")