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
class FrenchToEnglishDataset(Dataset):
    def __init__(self):
        # Reversed pairs (French to English)
        self.pairs = [
            ("J'ai froid", "I am cold"),
            ("Tu es fatigué", "You are tired"),
            ("Il a faim", "He is hungry"),
            ("Elle est heureuse", "She is happy"),
            ("Nous sommes amis", "We are friends"),
            ("Ils sont étudiants", "They are students"),
            ("Le chat dort", "The cat is sleeping"),
            ("Le soleil brille", "The sun is shining"),
            ("Nous aimons la musique", "We love music"),
            ("Elle parle français couramment", "She speaks French fluently"),
            ("Il aime lire des livres", "He enjoys reading books"),
            ("Ils jouent au football chaque week-end", "They play soccer every weekend"),
            ("Le film commence à 19 heures", "The movie starts at 7 PM"),
            ("Elle porte une robe rouge", "She wears a red dress"),
            ("Nous cuisinons le dîner ensemble", "We cook dinner together"),
            ("Il conduit une voiture bleue", "He drives a blue car"),
            ("Ils visitent souvent des musées", "They visit museums often"),
            ("Le restaurant sert une délicieuse cuisine", "The restaurant serves delicious food"),
            ("Elle étudie les mathématiques à l'université", "She studies mathematics at university"),
            ("Nous regardons des films le vendredi", "We watch movies on Fridays"),
            ("Il écoute de la musique en faisant du jogging", "He listens to music while jogging"),
            ("Ils voyagent autour du monde", "They travel around the world"),
            ("Le livre est sur la table", "The book is on the table"),
            ("Elle danse avec grâce", "She dances gracefully"),
            ("Nous célébrons les anniversaires avec un gâteau", "We celebrate birthdays with cake"),
            ("Il travaille dur tous les jours", "He works hard every day"),
            ("Ils parlent différentes langues", "They speak different languages"),
            ("Les fleurs fleurissent au printemps", "The flowers bloom in spring"),
            ("Elle écrit de la poésie pendant son temps libre", "She writes poetry in her free time"),
            ("Nous apprenons quelque chose de nouveau chaque jour", "We learn something new every day"),
            ("Le chien aboie bruyamment", "The dog barks loudly"),
            ("Il chante magnifiquement", "He sings beautifully"),
            ("Ils nagent dans la piscine", "They swim in the pool"),
            ("Les oiseaux gazouillent le matin", "The birds chirp in the morning"),
            ("Elle enseigne l'anglais à l'école", "She teaches English at school"),
            ("Nous prenons le petit déjeuner ensemble", "We eat breakfast together"),
            ("Il peint des paysages", "He paints landscapes"),
            ("Ils rient de la blague", "They laugh at the joke"),
            ("L'horloge tic-tac bruyamment", "The clock ticks loudly"),
            ("Elle court dans le parc", "She runs in the park"),
            ("Nous voyageons en train", "We travel by train"),
            ("Il écrit une lettre", "He writes a letter"),
            ("Ils lisent des livres à la bibliothèque", "They read books at the library"),
            ("Le bébé pleure", "The baby cries"),
            ("Elle étudie dur pour les examens", "She studies hard for exams"),
            ("Nous plantons des fleurs dans le jardin", "We plant flowers in the garden"),
            ("Il répare la voiture", "He fixes the car"),
            ("Ils boivent du café le matin", "They drink coffee in the morning"),
            ("Le soleil se couche le soir", "The sun sets in the evening"),
            ("Elle danse à la fête", "She dances at the party"),
            ("Nous jouons de la musique au concert", "We play music at the concert"),
            ("Il cuisine le dîner pour sa famille", "He cooks dinner for his family"),
            ("Ils étudient la grammaire française", "They study French grammar"),
            ("La pluie tombe doucement", "The rain falls gently"),
            ("Elle chante une chanson", "She sings a song"),
            ("Nous regardons un film ensemble", "We watch a movie together"),
            ("Il dort profondément", "He sleeps deeply"),
            ("Ils voyagent à Paris", "They travel to Paris"),
            ("Les enfants jouent dans le parc", "The children play in the park"),
            ("Elle se promène le long de la plage", "She walks along the beach"),
            ("Nous parlons au téléphone", "We talk on the phone"),
            ("Il attend le bus", "He waits for the bus"),
            ("Ils visitent la tour Eiffel", "They visit the Eiffel Tower"),
            ("Les étoiles scintillent la nuit", "The stars twinkle at night"),
            ("Elle rêve de voler", "She dreams of flying"),
            ("Nous travaillons au bureau", "We work in the office"),
            ("Il étudie l'histoire", "He studies history"),
            ("Ils écoutent la radio", "They listen to the radio"),
            ("Le vent souffle doucement", "The wind blows gently"),
            ("Elle nage dans l'océan", "She swims in the ocean"),
            ("Nous dansons au mariage", "We dance at the wedding"),
            ("Il gravit la montagne", "He climbs the mountain"),
            ("Ils font de la randonnée dans la forêt", "They hike in the forest"),
            ("Le chat miaule bruyamment", "The cat meows loudly"),
            ("Elle peint un tableau", "She paints a picture"),
            ("Nous construisons un château de sable", "We build a sandcastle"),
            ("Il chante dans le chœur", "He sings in the choir"),
            ("Ils font du vélo", "They ride bicycles"),
            ("Le café est chaud", "The coffee is hot"),
            ("Elle porte des lunettes", "She wears glasses"),
            ("Nous rendons visite à nos grands-parents", "We visit our grandparents"),
            ("Il joue de la guitare", "He plays the guitar"),
            ("Ils font du shopping", "They go shopping"),
            ("Le professeur explique la leçon", "The teacher explains the lesson"),
            ("Elle prend le train pour aller au travail", "She takes the train to work"),
            ("Nous faisons des biscuits", "We bake cookies"),
            ("Il se lave les mains", "He washes his hands"),
            ("Ils apprécient le coucher du soleil", "They enjoy the sunset"),
            ("La rivière coule calmement", "The river flows calmly"),
            ("Elle nourrit le chat", "She feeds the cat"),
            ("Nous visitons le musée", "We visit the museum"),
            ("Il répare son vélo", "He fixes his bicycle"),
            ("Ils peignent les murs", "They paint the walls"),
            ("Le bébé dort paisiblement", "The baby sleeps peacefully"),
            ("Elle attache ses lacets", "She ties her shoelaces"),
            ("Nous montons les escaliers", "We climb the stairs"),
            ("Il se rase le matin", "He shaves in the morning"),
            ("Ils mettent la table", "They set the table"),
            ("L'avion décolle", "The airplane takes off"),
            ("Elle arrose les plantes", "She waters the plants"),
            ("Nous pratiquons le yoga", "We practice yoga"),
            ("Il éteint la lumière", "He turns off the light"),
            ("Ils jouent aux jeux vidéo", "They play video games"),
            ("La soupe sent délicieusement bon", "The soup smells delicious"),
            ("Elle ferme la porte à clé", "She locks the door"),
            ("Nous profitons d'un pique-nique", "We enjoy a picnic"),
            ("Il vérifie ses emails", "He checks his email"),
            ("Ils vont à la salle de sport", "They go to the gym"),
            ("La lune brille intensément", "The moon shines brightly"),
            ("Elle attrape le bus", "She catches the bus"),
            ("Nous saluons nos voisins", "We greet our neighbors"),
            ("Il se peigne les cheveux", "He combs his hair"),
            ("Ils font un signe d'adieu", "They wave goodbye")
        ]
        # Create train/validation split
        self.train_pairs = self.pairs[:90]  # 90 training examples
        self.val_pairs = self.pairs[90:]    # 20 validation examples
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

# ====== Training + Evaluation =====
def train_model(model, train_loader, val_loader, vocab_size, epochs=20):
    # Move model to device
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train_losses, val_losses, val_accs = [], [], []
    
    model.train()
    for epoch in range(epochs):
        epoch_train_loss = 0
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

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                
                out = model(src, tgt)
                loss = criterion(out.reshape(-1, vocab_size), tgt[:,1:].reshape(-1))
                val_loss += loss.item()
                
                preds = out.argmax(-1)
                mask = (tgt[:,1:] != 0)  # Ignore padding tokens
                correct += ((preds == tgt[:,1:]) & mask).sum().item()
                total += mask.sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_acc = correct / total if total > 0 else 0
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        model.train()

    return train_losses, val_losses, val_accs

# ====== Qualitative Evaluation Function =====
def evaluate(model, phrase, dataset):
    model.eval()  # Set to evaluation mode
    return model.translate(phrase, dataset=dataset)

# ====== Main Script =====
if __name__ == "__main__":
    # Create dataset and loaders
    dataset = FrenchToEnglishDataset()
    vocab_size = len(dataset.vocab)
    
    # Split dataset into train and validation
    train_dataset = dataset
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
    val_loader = train_loader  # For simplicity, evaluate on training data
    
    results = {}
    models = {}  # Store models for qualitative evaluation

    # Print GPU memory usage before training
    if torch.cuda.is_available():
        print(f"GPU memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Train transformers with different configurations
    for layers in [1, 2, 4]:
        for heads in [2, 4]:
            label = f"Transformer ({layers}L-{heads}H)"
            print(f"\nTraining {label}...")
            model = TransformerSeq2Seq(vocab_size, num_layers=layers, nhead=heads)
            train_losses, val_losses, val_accs = train_model(model, train_loader, val_loader, vocab_size, epochs=20)
            results[label] = (train_losses, val_losses, val_accs)
            models[label] = model
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                print(f"GPU memory allocated after {label} training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Plot Results
    plt.figure(figsize=(20, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for k in results:
        plt.plot(results[k][0], label=k)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for k in results:
        plt.plot(results[k][1], label=k)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    for k in results:
        plt.plot(results[k][2], label=k)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("transformer_comparison.png")
    plt.show()

    # Print Final Results Table
    print("\nFinal Results:")
    print("{:<20} {:<15} {:<15} {:<15}".format("Model", "Train Loss", "Val Loss", "Val Acc"))
    print("-" * 65)
    for k, (train_loss, val_loss, val_acc) in results.items():
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.2f}%".format(
            k, train_loss[-1], val_loss[-1], val_acc[-1]*100))

    # Qualitative Evaluation
    print("\nQualitative Translation Examples:")
    test_phrases = [
        "J'ai froid",
        "Tu es fatigué",
        "Il a faim",
        "Elle parle français couramment",
        "Nous aimons la musique",
        "Le chat dort",
        "Ils visitent la tour Eiffel"
    ]

    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        for phrase in test_phrases:
            translated = evaluate(model, phrase, dataset)
            print(f"'{phrase}' -> '{translated}'")