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
    
    return X, y, vocab_size, char_to_int, int_to_char

class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# Model Architectures
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1), :]

class TransformerPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.model_type = 'transformer'
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final linear layer for prediction
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Embedding and positional encoding
        x = self.embed(x) * np.sqrt(self.embed.embedding_dim)  # Scale by sqrt(d_model)
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(x)
        
        # Get prediction for next character from the last token
        return self.fc(output[:, -1, :])

class TransformerWithCrossAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.model_type = 'transformer_with_cross'
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer decoder layer with cross-attention
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack encoder and decoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        # Query embedding (for cross-attention)
        self.query_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Final linear layer for prediction
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size = x.size(0)
        
        # Embedding and positional encoding
        x = self.embed(x) * np.sqrt(self.embed.embedding_dim)
        x = self.pos_encoder(x)
        
        # Encoder
        memory = self.transformer_encoder(x)
        
        # Prepare query for decoder (expand batch dimension)
        query = self.query_embed.expand(batch_size, -1, -1)
        
        # Decoder with cross-attention
        output = self.transformer_decoder(query, memory)
        
        # Get prediction from the decoder output
        return self.fc(output.squeeze(1))

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
    model_types = ['transformer', 'transformer_with_cross']
    all_results = []
    
    for seq_len in seq_lengths:
        X, y, vocab_size, _, _ = create_datasets(text, seq_len)
        dataset = TextDataset(X, y)
        
        # Train/Validation split
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, len(dataset)-train_size])
        
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        
        for model_type in model_types:
            print(f"\n=== Training {model_type.upper()} (seq_len={seq_len}) ===")
            
            if model_type == 'transformer':
                model = TransformerPredictor(
                    vocab_size=vocab_size,
                    embed_dim=64,
                    nhead=4,  # Number of attention heads
                    dim_feedforward=256,
                    num_layers=2
                ).to(device)
            elif model_type == 'transformer_with_cross':
                model = TransformerWithCrossAttention(
                    vocab_size=vocab_size,
                    embed_dim=64,
                    nhead=4,
                    dim_feedforward=256,
                    num_layers=2
                ).to(device)
            
            results = train_model(model, train_loader, val_loader, epochs=50)
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
        for model_type in ['transformer', 'transformer_with_cross']:
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
    print(f"{'Model':<20} {'Seq Len':<8} {'Loss':<8} {'Accuracy':<10} {'Time (s)':<10} {'Params':<10}")
    print("-" * 70)
    for res in sorted(results, key=lambda x: (x['seq_len'], x['model'])):
        model_name = res['model'].upper()
        if model_name == 'TRANSFORMER_WITH_CROSS':
            model_name = 'TRANS+CROSS'
        print(f"{model_name:<20} {res['seq_len']:<8} "
              f"{res['train_loss'][-1]:.4f}    {res['val_acc'][-1]:.4f}      "
              f"{res['time']:<10.1f} {res['params']:<10}")
    
    # Create a bar plot for comparison
    plt.figure(figsize=(14, 10))
    
    # Prepare data for bar plots
    models = ['TRANSFORMER', 'TRANS+CROSS']
    metrics = {
        'accuracy': {},
        'training_time': {},
        'parameters': {}
    }
    
    for seq_len in [10, 20, 30]:
        metrics['accuracy'][seq_len] = []
        metrics['training_time'][seq_len] = []
        metrics['parameters'][seq_len] = []
        
        for model_type in ['transformer', 'transformer_with_cross']:
            res = [r for r in results if r['seq_len'] == seq_len and r['model'] == model_type][0]
            metrics['accuracy'][seq_len].append(res['val_acc'][-1])
            metrics['training_time'][seq_len].append(res['time'])
            metrics['parameters'][seq_len].append(res['params'])
    
    # Plot metrics
    plt.subplot(3, 1, 1)
    index = np.arange(len(models))
    bar_width = 0.25
    for i, seq_len in enumerate([10, 20, 30]):
        plt.bar(index + i*bar_width, metrics['accuracy'][seq_len], bar_width, 
                label=f'Seq Len {seq_len}')
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(index + bar_width, models)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    for i, seq_len in enumerate([10, 20, 30]):
        plt.bar(index + i*bar_width, metrics['training_time'][seq_len], bar_width, 
                label=f'Seq Len {seq_len}')
    plt.xlabel('Model')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(index + bar_width, models)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    for i, seq_len in enumerate([10, 20, 30]):
        plt.bar(index + i*bar_width, 
                [p/1000 for p in metrics['parameters'][seq_len]], # Convert to thousands
                bar_width, label=f'Seq Len {seq_len}')
    plt.xlabel('Model')
    plt.ylabel('Number of Parameters (K)')
    plt.title('Model Size Comparison')
    plt.xticks(index + bar_width, models)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ======================
# Additional function for model inference
# ======================
def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, num_chars=100):
    model.eval()
    chars = []
    
    # Convert seed text to indices
    current_seq = np.array([char_to_int[ch] for ch in seed_text])
    
    for i in range(num_chars):
        # Convert to tensor and predict
        x = torch.tensor([current_seq[-seq_length:]], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = model(x)
        
        # Get next character
        _, next_idx = torch.max(pred, dim=1)
        next_char = int_to_char[next_idx.item()]
        chars.append(next_char)
        
        # Update sequence
        current_seq = np.append(current_seq, next_idx.item())
    
    return seed_text + ''.join(chars)

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    results = run_experiments()
    analyze_results(results)
    
    # Optional: Generate text with the best model
    # Pick the best model based on validation accuracy
    best_result = max(results, key=lambda x: x['val_acc'][-1])
    print(f"\nBest model: {best_result['model'].upper()} with sequence length {best_result['seq_len']}")
    
    # Re-create the best model and load with the best params
    seq_len = best_result['seq_len']
    X, y, vocab_size, char_to_int, int_to_char = create_datasets(text, seq_len)
    
    if best_result['model'] == 'transformer':
        best_model = TransformerPredictor(
            vocab_size=vocab_size,
            embed_dim=64,
            nhead=4,
            dim_feedforward=256,
            num_layers=2
        ).to(device)
    elif best_result['model'] == 'transformer_with_cross':
        best_model = TransformerWithCrossAttention(
            vocab_size=vocab_size,
            embed_dim=64,
            nhead=4,
            dim_feedforward=256,
            num_layers=2
        ).to(device)
        
    # Note: In a real scenario, you would save and load the model weights
    # Here, we're just recreating and could retrain if needed
    
    # Generate text example (commented out since we don't have actual trained weights)
    # seed = text[:100]  # Use first 100 chars as seed
    # generated = generate_text(best_model, seed, char_to_int, int_to_char, seq_len)
    # print(f"\nGenerated text sample:\n{generated}")