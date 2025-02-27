import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data Preparation
# ======================
# Your provided text sequence
text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

def create_datasets(text, seq_length):
    # Create character mappings
    chars = sorted(list(set(text)))
    char_to_int = {ch:i for i, ch in enumerate(chars)}
    int_to_char = {i:ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    # Encode text
    encoded = np.array([char_to_int[ch] for ch in text])
    
    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(len(encoded) - seq_length):
        sequences.append(encoded[i:i+seq_length])
        targets.append(encoded[i+seq_length])
    
    # Convert to tensors
    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    
    return X, y, vocab_size

class TextDataset(Dataset):
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
        
        if model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        elif model_type == 'lstm':
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
def train_model(model, train_loader, val_loader, epochs=50):
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
    seq_lengths = [10, 20, 30]
    model_types = ['rnn', 'lstm', 'gru']
    all_results = []
    
    for seq_len in seq_lengths:
        X, y, vocab_size = create_datasets(text, seq_len)
        dataset = TextDataset(X, y)
        
        # Train/Validation split
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, len(dataset)-train_size])
        
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        
        for model_type in model_types:
            print(f"\n=== Training {model_type.upper()} (seq_len={seq_len}) ===")
            
            model = CharPredictor(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_size=128,
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
# Visualization & Reporting
# ======================
def analyze_results(results):
    # Plot training curves
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, seq_len in enumerate([10, 20, 30]):
        for j, model_type in enumerate(['rnn', 'lstm', 'gru']):
            res = [r for r in results if r['seq_len'] == seq_len and r['model'] == model_type][0]
            axs[0,i].plot(res['train_loss'], label=model_type.upper())
            axs[1,i].plot(res['val_acc'], label=model_type.upper())
        
        axs[0,i].set_title(f'Seq Len {seq_len} - Training Loss')
        axs[1,i].set_title(f'Seq Len {seq_len} - Validation Accuracy')
        
    for ax in axs.flatten():
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    
    # Print final metrics
    print("\nFinal Metrics Comparison:")
    print(f"{'Model':<6} {'Seq Len':<8} {'Loss':<8} {'Accuracy':<10} {'Time (s)':<10} {'Params':<10}")
    for res in results:
        print(f"{res['model'].upper():<6} {res['seq_len']:<8} "
              f"{res['train_loss'][-1]:.4f}    {res['val_acc'][-1]:.4f}      "
              f"{res['time']:<10.1f} {res['params']:<10}")

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    results = run_experiments()
    analyze_results(results)
    plt.show()