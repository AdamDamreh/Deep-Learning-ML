import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random

# Reproducibility
random.seed(0)
torch.manual_seed(0)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Dataset =====
class ToyTranslationDataset(Dataset):
    def __init__(self):
        self.pairs = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur"),
    ("They ride bicycles", "Ils font du vélo"),
    ("The coffee is hot", "Le café est chaud"),
    ("She wears glasses", "Elle porte des lunettes"),
    ("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
    ("He plays the guitar", "Il joue de la guitare"),
    ("They go shopping", "Ils font du shopping"),
    ("The teacher explains the lesson", "Le professeur explique la leçon"),
    ("She takes the train to work", "Elle prend le train pour aller au travail"),
    ("We bake cookies", "Nous faisons des biscuits"),
    ("He washes his hands", "Il se lave les mains"),
    ("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
    ("The river flows calmly", "La rivière coule calmement"),
    ("She feeds the cat", "Elle nourrit le chat"),
    ("We visit the museum", "Nous visitons le musée"),
    ("He fixes his bicycle", "Il répare son vélo"),
    ("They paint the walls", "Ils peignent les murs"),
    ("The baby sleeps peacefully", "Le bébé dort paisiblement"),
    ("She ties her shoelaces", "Elle attache ses lacets"),
    ("We climb the stairs", "Nous montons les escaliers"),
    ("He shaves in the morning", "Il se rase le matin"),
    ("They set the table", "Ils mettent la table"),
    ("The airplane takes off", "L'avion décolle"),
    ("She waters the plants", "Elle arrose les plantes"),
    ("We practice yoga", "Nous pratiquons le yoga"),
    ("He turns off the light", "Il éteint la lumière"),
    ("They play video games", "Ils jouent aux jeux vidéo"),
    ("The soup smells delicious", "La soupe sent délicieusement bon"),
    ("She locks the door", "Elle ferme la porte à clé"),
    ("We enjoy a picnic", "Nous profitons d'un pique-nique"),
    ("He checks his email", "Il vérifie ses emails"),
    ("They go to the gym", "Ils vont à la salle de sport"),
    ("The moon shines brightly", "La lune brille intensément"),
    ("She catches the bus", "Elle attrape le bus"),
    ("We greet our neighbors", "Nous saluons nos voisins"),
    ("He combs his hair", "Il se peigne les cheveux"),
    ("They wave goodbye", "Ils font un signe d'adieu")
        ]
        self.vocab = self.build_vocab()

    def build_vocab(self):
        chars = set()
        for src, tgt in self.pairs:
            chars.update(src)
            chars.update(tgt)
        chars = sorted(list(chars.union({'<pad>', '<sos>', '<eos>'})))
        return {ch: i for i, ch in enumerate(chars)}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src = ['<sos>'] + list(src) + ['<eos>']
        tgt = ['<sos>'] + list(tgt) + ['<eos>']
        src_ids = [self.vocab[ch] for ch in src]
        tgt_ids = [self.vocab[ch] for ch in tgt]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# ====== Collate =====
def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, padding_value=0, batch_first=True)
    tgts = nn.utils.rnn.pad_sequence(tgts, padding_value=0, batch_first=True)
    return srcs, tgts

# ====== Transformer Model =====
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, num_layers=2, nhead=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.pos_enc = nn.Parameter(torch.randn(100, 64))
        self.transformer = nn.Transformer(d_model=64, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(64, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_enc[:src.size(1)]
        tgt_emb = self.embedding(tgt[:, :-1]) + self.pos_enc[:tgt.size(1)-1]
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt[:, :-1] == 0)
        output = self.transformer(src_emb.transpose(0, 1), tgt_emb.transpose(0, 1),
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc(output.transpose(0, 1))
    
    def translate(self, src, max_len=50, dataset=None):
        self.eval()
        index_to_char = {i: ch for ch, i in dataset.vocab.items()}
        
        src = torch.tensor([dataset.vocab.get(c, dataset.vocab['<sos>']) for c in ['<sos>'] + list(src) + ['<eos>']]).unsqueeze(0).to(device)
        src_emb = self.embedding(src) + self.pos_enc[:src.size(1)]
        src_key_padding_mask = (src == 0)
        
        # Initial decoder input (start token)
        out_seq = [dataset.vocab['<sos>']]
        
        for _ in range(max_len):
            tgt = torch.tensor([out_seq]).to(device)
            tgt_emb = self.embedding(tgt) + self.pos_enc[:tgt.size(1)]
            tgt_key_padding_mask = (tgt == 0)
            
            encoder_output = self.transformer.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
            output = self.transformer.decoder(tgt_emb.transpose(0, 1), encoder_output, tgt_key_padding_mask=tgt_key_padding_mask)
            output = self.fc(output[-1]).argmax(dim=1).item()
            
            if output == dataset.vocab['<eos>']:
                break
                
            out_seq.append(output)
        
        return ''.join([index_to_char[idx] for idx in out_seq[1:] if idx not in [dataset.vocab['<sos>'], dataset.vocab['<eos>'], dataset.vocab['<pad>']]])

# ====== RNN (Vanilla) =====
class RNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.encoder = nn.GRU(64, 128, batch_first=True)
        self.decoder = nn.GRU(64, 128, batch_first=True)
        self.out = nn.Linear(128, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, src, tgt):
        src_emb = self.embed(src)
        _, h = self.encoder(src_emb)
        tgt_emb = self.embed(tgt[:, :-1])
        output, _ = self.decoder(tgt_emb, h)
        return self.out(output)
    
    def translate(self, src, max_len=50, dataset=None):
        self.eval()
        index_to_char = {i: ch for ch, i in dataset.vocab.items()}
        
        src = torch.tensor([dataset.vocab.get(c, dataset.vocab['<sos>']) for c in ['<sos>'] + list(src) + ['<eos>']]).unsqueeze(0).to(device)
        src_emb = self.embed(src)
        _, h = self.encoder(src_emb)
        
        # Initial decoder input (start token)
        out_seq = [dataset.vocab['<sos>']]
        
        for _ in range(max_len):
            tgt = torch.tensor([[out_seq[-1]]]).to(device)
            tgt_emb = self.embed(tgt)
            output, h = self.decoder(tgt_emb, h)
            output = self.out(output).argmax(dim=2).item()
            
            if output == dataset.vocab['<eos>']:
                break
                
            out_seq.append(output)
        
        return ''.join([index_to_char[idx] for idx in out_seq[1:] if idx not in [dataset.vocab['<sos>'], dataset.vocab['<eos>'], dataset.vocab['<pad>']]])

# ====== RNN with Attention =====
class AttentionRNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.encoder = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(64 + 256, 128, batch_first=True)
        self.attn = nn.Linear(128 + 128 * 2, 1)
        self.fc = nn.Linear(128, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, src, tgt):
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt[:, :-1])
        encoder_outputs, _ = self.encoder(src_emb)
        h = torch.zeros(1, src.size(0), 128).to(src.device)
        outputs = []
        for t in range(tgt_emb.size(1)):
            repeat_h = h[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            attn_weights = self.attn(torch.cat((repeat_h, encoder_outputs), dim=2)).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            rnn_input = torch.cat((tgt_emb[:, t:t+1], context), dim=2)
            out, h = self.decoder(rnn_input, h)
            outputs.append(self.fc(out))
        return torch.cat(outputs, dim=1)
    
    def translate(self, src, max_len=50, dataset=None):
        self.eval()
        index_to_char = {i: ch for ch, i in dataset.vocab.items()}
        
        src = torch.tensor([dataset.vocab.get(c, dataset.vocab['<sos>']) for c in ['<sos>'] + list(src) + ['<eos>']]).unsqueeze(0).to(device)
        src_emb = self.embed(src)
        encoder_outputs, _ = self.encoder(src_emb)
        
        h = torch.zeros(1, 1, 128).to(device)
        out_seq = [dataset.vocab['<sos>']]
        
        for _ in range(max_len):
            tgt = torch.tensor([[out_seq[-1]]]).to(device)
            tgt_emb = self.embed(tgt)
            
            repeat_h = h[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            attn_weights = self.attn(torch.cat((repeat_h, encoder_outputs), dim=2)).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            
            rnn_input = torch.cat((tgt_emb, context), dim=2)
            out, h = self.decoder(rnn_input, h)
            output = self.fc(out).argmax(dim=2).item()
            
            if output == dataset.vocab['<eos>']:
                break
                
            out_seq.append(output)
        
        return ''.join([index_to_char[idx] for idx in out_seq[1:] if idx not in [dataset.vocab['<sos>'], dataset.vocab['<eos>'], dataset.vocab['<pad>']]])

# ====== Training + Evaluation =====
def train_model(model, train_loader, vocab_size):
    # Move model to device
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses, val_losses, accs = [], [], []
    train_losses = []  # Added to track training loss
    
    model.train()
    for epoch in range(50):
        epoch_train_loss = 0  # Track loss per epoch
        for src, tgt in train_loader:
            # Move data to device
            src = src.to(device)
            tgt = tgt.to(device)
            
            out = model(src, tgt)
            loss = criterion(out.reshape(-1, vocab_size), tgt[:,1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        # Average training loss for this epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(src, tgt)
            val_loss = criterion(out.reshape(-1, vocab_size), tgt[:,1:].reshape(-1)).item()
            preds = out.argmax(-1)
            correct = (preds == tgt[:,1:]).float()
            val_acc = correct.sum() / correct.numel()
            val_losses.append(val_loss)
            accs.append(val_acc.item())
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc.item()*100:.2f}%")
        model.train()

    return losses, val_losses, accs, train_losses

# ====== Qualitative Evaluation Function =====
def evaluate(model, phrase, dataset):
    model.eval()  # Set to evaluation mode
    return model.translate(phrase, dataset=dataset)

# ====== Main Comparison Loop =====
dataset = ToyTranslationDataset()
vocab_size = len(dataset.vocab)
loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

results = {}
models = {}  # Store models for qualitative evaluation

# Print GPU memory usage before training
if torch.cuda.is_available():
    print(f"GPU memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# RNN
print("Training Vanilla RNN...")
model = RNNSeq2Seq(vocab_size)
results['RNN'] = train_model(model, loader, vocab_size)
models['RNN'] = model

# Print GPU memory usage after RNN training
if torch.cuda.is_available():
    print(f"GPU memory allocated after RNN training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# RNN + Attention
print("Training RNN with Attention...")
model = AttentionRNNSeq2Seq(vocab_size)
results['RNN + Attention'] = train_model(model, loader, vocab_size)
models['RNN + Attention'] = model

# Print GPU memory usage after RNN+Attention training
if torch.cuda.is_available():
    print(f"GPU memory allocated after RNN+Attention training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Transformers
for layers in [1, 2, 4]:
    for heads in [2, 4]:
        label = f"Transformer ({layers}L-{heads}H)"
        print(f"Training {label}...")
        model = TransformerSeq2Seq(vocab_size, num_layers=layers, nhead=heads)
        results[label] = train_model(model, loader, vocab_size)
        models[label] = model

# Print final GPU memory usage
if torch.cuda.is_available():
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# ====== Plot Results =====
def plot_results():
    # Create a 2x2 figure for training loss, validation loss, validation accuracy
    plt.figure(figsize=(20, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for k in results:
        plt.plot(results[k][3], label=k)  # training loss
    plt.title("Training Loss")
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for k in results:
        plt.plot(results[k][1], label=k)  # validation loss
    plt.title("Validation Loss")
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    for k in results:
        plt.plot(results[k][2], label=k)  # validation accuracy
    plt.title("Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_results()

# ====== Print Final Table =====
print("\nFinal Results:")
print("Model\t\t\tTrain Loss\tVal Loss\tVal Acc")
for k, (_, val_loss, acc, train_loss) in results.items():
    print(f"{k:20s}\t{train_loss[-1]:.4f}\t{val_loss[-1]:.4f}\t{acc[-1]*100:.2f}%")

# ====== Qualitative Evaluation =====
print("\nQualitative examples:")
test_phrases = [
    "I am cold",
    "You are tired",
    "He is hungry",
]

# We'll test each model
for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    for phrase in test_phrases:
        translated = evaluate(model, phrase, dataset)
        print(f"'{phrase}' -> '{translated}'")