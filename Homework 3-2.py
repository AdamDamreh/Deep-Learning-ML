import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import time
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data Preparation
# ======================
def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return requests.get(url).text

def create_datasets(text, seq_length):
    chars = sorted(list(set(text)))
    char_to_int = {ch:i for i, ch in enumerate(chars)}
    encoded = np.array([char_to_int[ch] for ch in text])
    
    # Calculate number of sequences
    num_sequences = len(encoded) - seq_length
    
    # Pre-allocate numpy arrays
    sequences = np.zeros((num_sequences, seq_length), dtype=np.int64)
    targets = np.zeros(num_sequences, dtype=np.int64)
    
    # Fill arrays using sliding window
    for i in range(num_sequences):
        sequences[i] = encoded[i:i+seq_length]
        targets[i] = encoded[i+seq_length]
    
    # Convert to tensors in one operation
    return (
        torch.from_numpy(sequences),
        torch.from_numpy(targets),
        len(chars),
        char_to_int
    )

class ShakespeareDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# Model Architecture
# ======================
class CharPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, model_type):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# ======================
# Training Infrastructure
# ======================
def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    results = {'train_loss': [], 'val_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        results['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        results['val_acc'].append(correct/total)
        print(f'Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {correct/total:.4f}')
    
    results['time'] = time.time() - start_time
    results['params'] = sum(p.numel() for p in model.parameters())
    return results

# ======================
# Experiment Runner
# ======================
def run_experiments():
    text = load_shakespeare()
    seq_lengths = [20, 30, 50]

    model_types = ['lstm', 'gru']
    all_results = []
    
    for seq_len in seq_lengths:
        X, y, vocab_size, _ = create_datasets(text, seq_len)
        dataset = ShakespeareDataset(X, y)
        
        # 80-20 split
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, len(dataset)-train_size])
        
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        
        for model_type in model_types:
            print(f"\n=== Training {model_type.upper()} (seq_len={seq_len}) ===")
            
            model = CharPredictor(
                vocab_size=vocab_size,
                embed_dim=128,
                hidden_size=256,
                model_type=model_type
            ).to(device)
            
            results = train_model(model, train_loader, val_loader)
            all_results.append({
                'model': model_type,
                'seq_len': seq_len,
                **results
            })
    
    return all_results

# ======================
# Visualization
# ======================
def plot_comparison(results):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training Time
    for model in ['lstm', 'gru']:
        times = [r['time'] for r in results if r['model'] == model]
        seq_lens = [r['seq_len'] for r in results if r['model'] == model]
        axs[0].plot(seq_lens, times, 'o-', label=model.upper())
    axs[0].set_title('Training Time Comparison')
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Time (seconds)')
    axs[0].legend()
    
    # Validation Accuracy
    for model in ['lstm', 'gru']:
        accs = [r['val_acc'][-1] for r in results if r['model'] == model]
        seq_lens = [r['seq_len'] for r in results if r['model'] == model]
        axs[1].plot(seq_lens, accs, 'o-', label=model.upper())
    axs[1].set_title('Final Validation Accuracy Comparison')
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    results = run_experiments()
    plot_comparison(results)
    
    # Print final metrics
    print("\nFinal Results Summary:")
    for res in results:
        print(f"{res['model'].upper()} ({res['seq_len']}): "
              f"Loss={res['train_loss'][-1]:.4f}, Acc={res['val_acc'][-1]:.4f}, "
              f"Time={res['time']:.1f}s, Params={res['params']}")