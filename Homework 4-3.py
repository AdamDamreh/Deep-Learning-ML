import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
french_to_english = [
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

# Special tokens
SOS_token = 0
EOS_token = 1

# Create character mappings
all_chars = set()
for fr, eng in french_to_english:
    all_chars.update(eng)
    all_chars.update(fr)

char_to_index = {"SOS": SOS_token, "EOS": EOS_token}
index_to_char = {SOS_token: "SOS", EOS_token: "EOS"}

for i, char in enumerate(sorted(all_chars)):
    char_to_index[char] = i + 2
    index_to_char[i + 2] = char

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, pairs, char_to_index):
        self.pairs = pairs
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_sentence, target_sentence = self.pairs[idx]
        input_indices = [self.char_to_index[c] for c in input_sentence] + [EOS_token]
        target_indices = [self.char_to_index[c] for c in target_sentence] + [EOS_token]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

# GRU Models
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Training function
def train(input_tensor, target_tensor, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion, max_length=50):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    # Encoder forward
    for ei in range(input_length):
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # Decoder forward
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Validation loss calculation
def calculate_validation_loss(encoder, decoder, dataloader, criterion):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor = input_tensor.squeeze().to(device)
            target_tensor = target_tensor.squeeze().to(device)
            
            encoder_hidden = encoder.initHidden()
            loss = 0
            
            # Encoder
            for ei in range(input_tensor.size(0)):
                _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            
            # Decoder
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            
            for di in range(target_tensor.size(0)):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break
            
            total_loss += loss.item() / target_tensor.size(0)
    
    return total_loss / len(dataloader)


# Evaluation function
def evaluate(encoder, decoder, sentence, max_length=50):
    with torch.no_grad():
        input_tensor = torch.tensor([char_to_index[c] for c in sentence] + [EOS_token], device=device)
        input_length = input_tensor.size(0)
        
        encoder_hidden = encoder.initHidden()
        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        
        decoded_chars = []
        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            
            if topi.item() == EOS_token:
                break
                
            decoded_chars.append(index_to_char[topi.item()])
            decoder_input = topi.squeeze()
        
        return ''.join(decoded_chars)

# Training setup
hidden_size = 256
input_size = len(char_to_index)
output_size = len(char_to_index)
learning_rate = 0.005
n_epochs = 100

encoder = EncoderGRU(input_size, hidden_size).to(device)
decoder = DecoderGRU(hidden_size, output_size).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

dataset = TranslationDataset(french_to_english, char_to_index)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        input_tensor = input_tensor.squeeze().to(device)
        target_tensor = target_tensor.squeeze().to(device)
        
        loss = train(input_tensor, target_tensor, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss
    
    if epoch % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Evaluation
encoder.eval()
decoder.eval()

total = 0
correct = 0

with torch.no_grad():
    for fr, eng in french_to_english:
        translated = evaluate(encoder, decoder, fr)
        print(f"Input: {fr}")
        print(f"Target: {eng}")
        print(f"Translation: {translated}\n")
        
        if translated == eng:
            correct += 1
        total += 1

print(f"Validation Accuracy: {correct/total:.2%}")
validation_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
validation_loss = calculate_validation_loss(encoder, decoder, validation_dataloader, criterion)
print(f"Validation Loss: {validation_loss:.4f}")

# Qualitative examples
test_phrases = [
    "J'ai froid",
    "Tu es fatigué",
    "Il a faim",
]
for phrase in test_phrases:
    translated = evaluate(encoder, decoder, phrase)
    print(f"'{phrase}' -> '{translated}'")