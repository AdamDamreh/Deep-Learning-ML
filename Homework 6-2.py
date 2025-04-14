import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from transformers import SwinForImageClassification, AutoImageProcessor
import numpy as np
from torchinfo import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters for fine-tuning
num_classes = 100  # CIFAR-100
batch_size = 32
learning_rate = 2e-5
num_epochs = 3  # We'll use 3 epochs as a balance for the fine-tuning example

# Load pretrained image processor for Swin Transformer
tiny_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
small_processor = AutoImageProcessor.from_pretrained("microsoft/swin-small-patch4-window7-224")

# Define transforms for CIFAR-100
def get_transforms(processor):
    # Define transforms that match the preprocessing requirements of Swin models
    # Swin models expect 224x224 images
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize to slightly larger
        transforms.CenterCrop(224),  # Then center crop to expected size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean, 
            std=processor.image_std
        )
    ])
    
    # Add augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean, 
            std=processor.image_std
        )
    ])
    
    return train_transform, preprocess

# Data preparation
def load_datasets(train_transform, test_transform):
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Load and modify Swin Transformer models
def load_swin_model(model_name, num_classes=100):
    # Load pretrained model
    model = SwinForImageClassification.from_pretrained(model_name)
    
    # Replace classification head for CIFAR-100
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # Freeze backbone parameters
    for param in model.swin.parameters():
        param.requires_grad = False
    
    # Only classifier parameters will be trained
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    return model

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train function for fine-tuning
def train_model(model, model_name, train_loader, test_loader, num_epochs=num_epochs):
    model = model.to(device)
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
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
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
                outputs = model(images).logits
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

# Main execution function
def experiment():
    results = {}
    
    # Prepare Swin Tiny
    print("\nPreparing Swin Tiny Transformer...")
    tiny_train_transform, tiny_test_transform = get_transforms(tiny_processor)
    tiny_train_loader, tiny_test_loader = load_datasets(tiny_train_transform, tiny_test_transform)
    tiny_model = load_swin_model("microsoft/swin-tiny-patch4-window7-224", num_classes)
    
    # Prepare Swin Small
    print("\nPreparing Swin Small Transformer...")
    small_train_transform, small_test_transform = get_transforms(small_processor)
    small_train_loader, small_test_loader = load_datasets(small_train_transform, small_test_transform)
    small_model = load_swin_model("microsoft/swin-small-patch4-window7-224", num_classes)
    
    # Fine-tune Swin Tiny
    print("\nFine-tuning Swin Tiny Transformer...")
    tiny_stats = train_model(tiny_model, "Swin-Tiny", tiny_train_loader, tiny_test_loader)
    results["Swin-Tiny (Fine-tuned)"] = tiny_stats
    
    # Fine-tune Swin Small
    print("\nFine-tuning Swin Small Transformer...")
    small_stats = train_model(small_model, "Swin-Small", small_train_loader, small_test_loader)
    results["Swin-Small (Fine-tuned)"] = small_stats
    
    # Summarize results
    print("\n===== Fine-tuning Results Summary =====")
    headers = ["Model", "Test Acc.", "Trainable Params (M)", "Time/Epoch (s)"]
    print(f"{headers[0]:<25} {headers[1]:<10} {headers[2]:<20} {headers[3]:<15}")
    print("-" * 75)
    
    for model_name, stats in results.items():
        acc = stats['final_accuracy']
        params = stats['parameters'] / 1e6  # Convert to millions
        time_per_epoch = stats['avg_epoch_time']
        
        print(f"{model_name:<25} {acc:<10.2f} {params:<20.2f} {time_per_epoch:<15.2f}")
    
    # Compare with training from scratch
    print("\n===== Comparison with Training from Scratch =====")
    print("Model                     | Test Acc. | Params (M) | Training Time")
    print("--------------------------|-----------|------------|-------------")
    print("Swin-Tiny (Fine-tuned)    | {:5.2f}    | {:5.2f}      | Fast (few epochs)".format(
        results["Swin-Tiny (Fine-tuned)"]["final_accuracy"], 
        results["Swin-Tiny (Fine-tuned)"]["parameters"] / 1e6
    ))
    print("Swin-Small (Fine-tuned)   | {:5.2f}    | {:5.2f}      | Fast (few epochs)".format(
        results["Swin-Small (Fine-tuned)"]["final_accuracy"], 
        results["Swin-Small (Fine-tuned)"]["parameters"] / 1e6
    ))
    print("ViT-Tiny (From Scratch)   | ~20.00    | ~5.00      | Slow (many epochs)")
    print("ViT-Small (From Scratch)  | < 10.00   | ~17.50     | Very slow (many epochs)")
    
    return results

# Estimate FLOPs using torchinfo (simplified for demonstration)
def estimate_model_complexity():
    print("\n===== Model Complexity Analysis =====")
    
    # Load models for analysis only
    tiny_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    small_model = SwinForImageClassification.from_pretrained("microsoft/swin-small-patch4-window7-224")
    
    # Print basic model info
    tiny_params = sum(p.numel() for p in tiny_model.parameters()) / 1e6
    small_params = sum(p.numel() for p in small_model.parameters()) / 1e6
    
    print(f"Swin-Tiny - Total parameters: {tiny_params:.2f}M")
    print(f"Swin-Small - Total parameters: {small_params:.2f}M")
    
    # For full FLOPs analysis, would use torchinfo
    # Simplified version for demonstration
    print("\nApproximate GFLOPs (224x224 image):")
    print("Swin-Tiny: ~4.5 GFLOPs")
    print("Swin-Small: ~8.7 GFLOPs")

# Run the experiments
if __name__ == '__main__':
    # First, analyze model complexity
    estimate_model_complexity()
    
    # Then run fine-tuning experiments
    results = experiment()
    
    # Discussion
    print("\n===== Discussion =====")
    print("Fine-tuning vs Training from Scratch:")
    print("1. Fine-tuning pretrained models achieves higher accuracy with much less training time")
    print("2. Only the classification head is trained, reducing computational requirements")
    print("3. Pretrained models benefit from knowledge transfer from larger datasets (ImageNet)")
    print("4. Need to resize CIFAR-100 images from 32x32 to 224x224, potentially losing detail")
    
    print("\nSwin-Tiny vs Swin-Small:")
    print("1. Swin-Small has more parameters and computational requirements")
    print("2. Swin-Small may achieve higher accuracy due to increased capacity")
    print("3. Swin-Tiny offers better efficiency for similar performance on simpler tasks")
    
    print("\nReasons for Performance Differences:")
    print("1. Pretrained models have learned general visual features from ImageNet (1.2M images)")
    print("2. Hierarchical structure of Swin Transformers helps with different scales of features")
    print("3. Custom models may be better optimized for specific dataset characteristics")
    print("4. The shift-window attention mechanism in Swin models improves efficiency")