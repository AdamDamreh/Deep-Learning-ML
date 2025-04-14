import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
from torchinfo import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_classes = 100  # CIFAR-100
num_epochs = 20
batch_size = 64
learning_rate = 0.001

# Data preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

# CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x2 = self.layer_norm1(x)
        attention_output, _ = self.attention(x2, x2, x2)
        x = x + attention_output
        x2 = self.layer_norm2(x)
        mlp_output = self.mlp(x2)
        x = x + mlp_output
        return x

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=100, embed_dim=256, 
                 num_heads=4, num_layers=4, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for transformer in self.transformer:
            x = transformer(x)
            
        x = self.layer_norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train function
def train_model(model, model_name, num_epochs=num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # For storing metrics
    train_losses = []
    test_accuracies = []
    epoch_times = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f'{model_name} - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Test accuracy after each epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(f'{model_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'epoch_times': epoch_times,
        'final_accuracy': test_accuracies[-1],
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'parameters': count_parameters(model)
    }

# Function to estimate FLOPs using torchinfo
def estimate_flops(model, input_size=(1, 3, 32, 32)):
    model_info = summary(model, input_size=input_size, verbose=0)
    return model_info.total_mult_adds

# Define configurations for experimentation
configurations = [
    {
        'name': 'ViT-Tiny',
        'params': {
            'patch_size': 4,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 2,
            'mlp_dim': 1024
        }
    },
    {
        'name': 'ViT-Small',
        'params': {
            'patch_size': 4,
            'embed_dim': 512,
            'num_layers': 4,
            'num_heads': 4, 
            'mlp_dim': 2048
        }
    },
    {
        'name': 'ViT-Medium',
        'params': {
            'patch_size': 8,
            'embed_dim': 256,
            'num_layers': 8,
            'num_heads': 4,
            'mlp_dim': 1024
        }
    },
    {
        'name': 'ViT-Large',
        'params': {
            'patch_size': 8,
            'embed_dim': 512,
            'num_layers': 8,
            'num_heads': 4,
            'mlp_dim': 2048
        }
    }
]

def create_resnet_baseline():
    # Load pretrained ResNet-18 and modify for CIFAR-100
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Main execution function
def experiment():
    results = {}
    
    # Train and evaluate different ViT configurations
    for config in configurations:
        print(f"\nTraining {config['name']}...")
        model = VisionTransformer(
            image_size=32,
            patch_size=config['params']['patch_size'],
            num_classes=num_classes,
            embed_dim=config['params']['embed_dim'],
            num_heads=config['params']['num_heads'],
            num_layers=config['params']['num_layers'],
            mlp_dim=config['params']['mlp_dim']
        ).to(device)
        
        # Calculate FLOPs
        flops = estimate_flops(model)
        
        # Train model
        training_stats = train_model(model, config['name'], num_epochs=20)  # Using only 10 epochs for comparison
        training_stats['flops'] = flops
        results[config['name']] = training_stats
    
    # Train and evaluate ResNet-18 baseline
    print("\nTraining ResNet-18 baseline...")
    resnet_model = create_resnet_baseline().to(device)
    resnet_flops = estimate_flops(resnet_model)
    resnet_stats = train_model(resnet_model, "ResNet-18", num_epochs=20)  # Using only 10 epochs for comparison
    resnet_stats['flops'] = resnet_flops
    results["ResNet-18"] = resnet_stats
    
    # Summarize results
    print("\n===== Results Summary =====")
    headers = ["Model", "Test Acc.", "Params (M)", "GFLOPs", "Time/Epoch (s)"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<12} {headers[3]:<10} {headers[4]:<15}")
    print("-" * 65)
    
    for model_name, stats in results.items():
        acc = stats['final_accuracy']
        params = stats['parameters'] / 1e6  # Convert to millions
        gflops = stats['flops'] / 1e9  # Convert to GFLOPs
        time_per_epoch = stats['avg_epoch_time']
        
        print(f"{model_name:<15} {acc:<10.2f} {params:<12.2f} {gflops:<10.2f} {time_per_epoch:<15.2f}")
    
    # Plot results
    plot_results(results)
    
    return results

def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for model_name, stats in results.items():
        plt.plot(stats['train_losses'], label=model_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(2, 2, 2)
    for model_name, stats in results.items():
        plt.plot(stats['test_accuracies'], label=model_name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot parameters vs accuracy
    plt.subplot(2, 2, 3)
    x = [stats['parameters'] / 1e6 for stats in results.values()]
    y = [stats['final_accuracy'] for stats in results.values()]
    labels = list(results.keys())
    plt.scatter(x, y)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]))
    plt.title('Parameters vs Accuracy')
    plt.xlabel('Parameters (M)')
    plt.ylabel('Accuracy (%)')
    
    # Plot GFLOPs vs accuracy
    plt.subplot(2, 2, 4)
    x = [stats['flops'] / 1e9 for stats in results.values()]
    y = [stats['final_accuracy'] for stats in results.values()]
    plt.scatter(x, y)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]))
    plt.title('GFLOPs vs Accuracy')
    plt.xlabel('GFLOPs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('vit_vs_resnet_results.png')
    plt.show()

# Run the experiment
if __name__ == '__main__':
    results = experiment()
    
    # Additional analysis and discussion can be added here
    print("\nAnalysis:")
    print("1. The ViT models have different computational profiles compared to ResNet-18.")
    print("2. Larger embedding dimensions and more layers generally lead to higher accuracy but increased computation.")
    print("3. Patch size significantly impacts the number of tokens and therefore computation time.")
    
    # You could uncomment this to save the model if needed
    # torch.save(best_model.state_dict(), 'best_vit_model.pth')