import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Step 2: Prepare the dataset
def create_dataset(text, sequence_length):
    # Create a character mapping to integers
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]
    
    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return sequences, targets, vocab_size, chars, char_to_int, int_to_char

# Step 3: Create a dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.output = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        
        # Create src mask (to prevent attending to padding tokens)
        src_mask = None  # No masking for character-level modeling
        
        # Convert token indices to embeddings and apply positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)  # [batch_size, seq_len, d_model]
        
        # Transpose for transformer input [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask)  # [seq_len, batch_size, d_model]
        
        # Take the last token's output for prediction
        output = output[-1]  # [batch_size, d_model]
        
        # Project to vocabulary size
        output = self.output(output)  # [batch_size, vocab_size]
        
        return output

# Simple RNN Model for comparison
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size = x.size(0)
        
        # Initialize hidden state
        hidden = self.init_hidden(batch_size).to(x.device)
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        
        # RNN
        out, hidden = self.rnn(x, hidden)  # out: [batch_size, seq_len, hidden_size]
        
        # Take the output from the last time step
        out = out[:, -1, :]  # [batch_size, hidden_size]
        
        # Linear layer
        out = self.fc(out)  # [batch_size, vocab_size]
        
        return out
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device, model_type="Transformer"):
    model.to(device)
    
    train_losses = []
    valid_accs = []
    times = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        valid_accs.append(accuracy)
        
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        scheduler.step()
        
        print(f'{model_type} - Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, '
              f'Valid Acc: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
    
    # Calculate total training time
    total_time = sum(times)
    print(f'Total training time: {total_time:.2f} seconds')
    
    return {
        'train_losses': train_losses,
        'valid_accs': valid_accs,
        'times': times,
        'total_time': total_time
    }

# Model complexity calculation
def calculate_model_complexity(model):
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate model size in MB
    model_size_bytes = total_params * 4  # Assuming 4 bytes per parameter (float32)
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    return {
        'params': total_params,
        'size_mb': model_size_mb
    }

# Generate text function - FIXED VERSION
def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, num_chars=100, device='cuda'):
    model.eval()
    
    # Convert seed text to tensor
    chars = [char_to_int[ch] for ch in seed_text]
    current_seq = torch.tensor([chars], dtype=torch.long).to(device)
    
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(num_chars):
            # Get the last sequence_length characters
            if current_seq.size(1) > seq_length:
                current_input = current_seq[:, -seq_length:]
            else:
                current_input = current_seq
                
            # Generate prediction
            output = model(current_input)
            
            # Get the most probable next character
            _, top_idx = torch.topk(output, 3)
            # Fix: Make sure choice is a scalar tensor
            choice_idx = torch.randint(0, 3, (1,)).item()
            choice = top_idx[0][choice_idx]
            
            # Add predicted character to sequence and generated text
            # Fix: Make sure the dimensions match
            choice = choice.view(1, 1)  # Reshape to [1, 1]
            current_seq = torch.cat((current_seq, choice), dim=1)
            generated_text += int_to_char[choice.item()]
    
    return generated_text

# Run experiments for different sequence lengths
def run_experiments(sequence_lengths, d_model=128, nhead=2, num_layers=2, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*50}")
        print(f"Running experiment with sequence length: {seq_len}")
        print(f"{'='*50}")
        
        # Create dataset
        sequences, targets, vocab_size, chars, char_to_int, int_to_char = create_dataset(text, seq_len)
        
        # Create DataLoader
        dataset = CharDataset(sequences, targets)
        batch_size = 128
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        
        # Create models
        transformer = TransformerModel(vocab_size, d_model, nhead, num_layers)
        rnn = RNNModel(vocab_size, d_model, num_layers)
        
        # Calculate model complexity
        transformer_complexity = calculate_model_complexity(transformer)
        rnn_complexity = calculate_model_complexity(rnn)
        
        print(f"Transformer complexity: {transformer_complexity['params']} parameters, {transformer_complexity['size_mb']:.2f} MB")
        print(f"RNN complexity: {rnn_complexity['params']} parameters, {rnn_complexity['size_mb']:.2f} MB")
        
        # Train transformer
        print("\nTraining Transformer model...")
        transformer_optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
        transformer_scheduler = torch.optim.lr_scheduler.StepLR(transformer_optimizer, step_size=5, gamma=0.5)
        transformer_criterion = nn.CrossEntropyLoss()
        
        transformer_results = train_model(
            transformer, train_loader, test_loader,
            transformer_criterion, transformer_optimizer, transformer_scheduler,
            epochs, device, "Transformer"
        )
        
        # Train RNN
        print("\nTraining RNN model...")
        rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
        rnn_scheduler = torch.optim.lr_scheduler.StepLR(rnn_optimizer, step_size=5, gamma=0.5)
        rnn_criterion = nn.CrossEntropyLoss()
        
        rnn_results = train_model(
            rnn, train_loader, test_loader,
            rnn_criterion, rnn_optimizer, rnn_scheduler,
            epochs, device, "RNN"
        )
        
        # Generate sample text
        seed_text = "ROMEO: "
        transformer_generated = generate_text(transformer, seed_text, char_to_int, int_to_char, seq_len, 100, device)
        rnn_generated = generate_text(rnn, seed_text, char_to_int, int_to_char, seq_len, 100, device)
        
        print("\nTransformer generated text:")
        print(transformer_generated)
        print("\nRNN generated text:")
        print(rnn_generated)
        
        # Store results
        results[seq_len] = {
            'transformer': {
                'training_results': transformer_results,
                'complexity': transformer_complexity,
                'generated_text': transformer_generated
            },
            'rnn': {
                'training_results': rnn_results,
                'complexity': rnn_complexity,
                'generated_text': rnn_generated
            }
        }
    
    return results, chars, char_to_int, int_to_char, vocab_size

# Hyperparameter exploration
def explore_hyperparameters(seq_length, layers_options, heads_options, d_model_options, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    sequences, targets, vocab_size, chars, char_to_int, int_to_char = create_dataset(text, seq_length)
    
    # Create DataLoader
    dataset = CharDataset(sequences, targets)
    batch_size = 128
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    results = {}
    
    for num_layers in layers_options:
        for nhead in heads_options:
            for d_model in d_model_options:
                config_name = f"L{num_layers}_H{nhead}_D{d_model}"
                print(f"\n{'='*50}")
                print(f"Training with config: {config_name}")
                print(f"Layers: {num_layers}, Heads: {nhead}, Hidden size: {d_model}")
                print(f"{'='*50}")
                
                # Create model
                model = TransformerModel(vocab_size, d_model, nhead, num_layers)
                
                # Calculate model complexity
                complexity = calculate_model_complexity(model)
                print(f"Model complexity: {complexity['params']} parameters, {complexity['size_mb']:.2f} MB")
                
                # Train model
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
                criterion = nn.CrossEntropyLoss()
                
                training_results = train_model(
                    model, train_loader, test_loader,
                    criterion, optimizer, scheduler,
                    epochs, device, config_name
                )
                
                # Generate sample text
                seed_text = "ROMEO: "
                generated_text = generate_text(model, seed_text, char_to_int, int_to_char, seq_length, 100, device)
                
                print("\nGenerated text:")
                print(generated_text)
                
                # Calculate inference time
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):  # Average over 10 runs
                        _ = generate_text(model, seed_text, char_to_int, int_to_char, seq_length, 20, device)
                inference_time = (time.time() - start_time) / 10
                
                print(f"Average inference time: {inference_time:.4f} seconds")
                
                # Store results
                results[config_name] = {
                    'training_results': training_results,
                    'complexity': complexity,
                    'generated_text': generated_text,
                    'inference_time': inference_time,
                    'config': {
                        'num_layers': num_layers,
                        'nhead': nhead,
                        'd_model': d_model
                    }
                }
    
    return results

# Main execution
if __name__ == "__main__":
    # Part 1: Compare different sequence lengths
    seq_lengths = [20, 30]
    part1_results, chars, char_to_int, int_to_char, vocab_size = run_experiments(seq_lengths)
    
    # Part 2: Hyperparameter exploration
    layers_options = [1, 2, 4]
    heads_options = [2, 4]
    d_model_options = [128]  # Fixed hidden size for simplicity
    part2_results = explore_hyperparameters(30, layers_options, heads_options, d_model_options)
    
    # Part 3: Increase sequence length to 50
    part3_results, _, _, _, _ = run_experiments([50])
    
    # Print summary
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    
    # Part 1 summary
    print("\nPart 1: Sequence Length Comparison")
    for seq_len in seq_lengths:
        tf_result = part1_results[seq_len]['transformer']
        rnn_result = part1_results[seq_len]['rnn']
        
        print(f"\nSequence Length: {seq_len}")
        print(f"Transformer final loss: {tf_result['training_results']['train_losses'][-1]:.4f}")
        print(f"Transformer final accuracy: {tf_result['training_results']['valid_accs'][-1]:.2f}%")
        print(f"Transformer training time: {tf_result['training_results']['total_time']:.2f}s")
        print(f"Transformer model size: {tf_result['complexity']['params']} params, {tf_result['complexity']['size_mb']:.2f} MB")
        
        print(f"RNN final loss: {rnn_result['training_results']['train_losses'][-1]:.4f}")
        print(f"RNN final accuracy: {rnn_result['training_results']['valid_accs'][-1]:.2f}%")
        print(f"RNN training time: {rnn_result['training_results']['total_time']:.2f}s")
        print(f"RNN model size: {rnn_result['complexity']['params']} params, {rnn_result['complexity']['size_mb']:.2f} MB")
    
    # Part 2 summary
    print("\nPart 2: Hyperparameter Exploration")
    for config_name, result in part2_results.items():
        config = result['config']
        print(f"\nConfig: {config_name}")
        print(f"Layers: {config['num_layers']}, Heads: {config['nhead']}, Hidden size: {config['d_model']}")
        print(f"Final loss: {result['training_results']['train_losses'][-1]:.4f}")
        print(f"Final accuracy: {result['training_results']['valid_accs'][-1]:.2f}%")
        print(f"Training time: {result['training_results']['total_time']:.2f}s")
        print(f"Inference time: {result['inference_time']:.4f}s")
        print(f"Model size: {result['complexity']['params']} params, {result['complexity']['size_mb']:.2f} MB")
    
    # Part 3 summary
    print("\nPart 3: Increased Sequence Length (50)")
    tf_result = part3_results[50]['transformer']
    rnn_result = part3_results[50]['rnn']
    
    print(f"Transformer final loss: {tf_result['training_results']['train_losses'][-1]:.4f}")
    print(f"Transformer final accuracy: {tf_result['training_results']['valid_accs'][-1]:.2f}%")
    print(f"Transformer training time: {tf_result['training_results']['total_time']:.2f}s")
    print(f"Transformer model size: {tf_result['complexity']['params']} params, {tf_result['complexity']['size_mb']:.2f} MB")
    
    print(f"RNN final loss: {rnn_result['training_results']['train_losses'][-1]:.4f}")
    print(f"RNN final accuracy: {rnn_result['training_results']['valid_accs'][-1]:.2f}%")
    print(f"RNN training time: {rnn_result['training_results']['total_time']:.2f}s")
    print(f"RNN model size: {rnn_result['complexity']['params']} params, {rnn_result['complexity']['size_mb']:.2f} MB")