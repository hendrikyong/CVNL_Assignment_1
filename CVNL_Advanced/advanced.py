import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

def text_to_sequence(text, vocab, max_len=10):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in word_tokenize(text.lower())]
    return tokens[:max_len] + [vocab["<PAD>"]] * (max_len - len(tokens))

class IntentBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(IntentBiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        x_embedded = self.embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(x_packed)
        hidden = hidden.view(self.rnn.num_layers, 2, x.size(0), self.rnn.hidden_size)
        hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        return self.fc(hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.pickle", "rb") as f:
    vocab = pickle.load(f)
with open("intent_labels.pickle", "rb") as f:
    intent_labels = pickle.load(f)

# reverse mapping from index to intent 
idx_to_intent = {idx: intent for intent, idx in intent_labels.items()}

# model parameters (used same as rnn model)
vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 64
output_dim = len(intent_labels)
max_len = 10  

model = IntentBiRNN(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("CVNL_RNN.pth", map_location=device))
model.to(device)
model.eval()

def predict_intent(text):
    tokenized = text_to_sequence(text, vocab, max_len)
    input_tensor = torch.tensor(tokenized).unsqueeze(0).to(device)
    length = torch.tensor([min(len(word_tokenize(text.lower())), max_len)], dtype=torch.int64).cpu()
    
    with torch.no_grad():
        output = model(input_tensor, length)
        predicted_idx = torch.argmax(output, dim=1).item()
    
    return idx_to_intent.get(predicted_idx, "Unknown")

def main():
    print("=== Intent-based Detection and Classification Tool ===")
    print("Available intents:")
    for intent in idx_to_intent.values():
        print(" -", intent)
    print("\nType your text below to detect its intent. Type 'exit' to quit.\n")
    
    while True:
        text = input("Enter text: ")
        if text.lower() == "exit":
            print("Exiting the tool. Goodbye!")
            break
        
        intent = predict_intent(text)
        print("Predicted Intent:", intent, "\n")

if __name__ == "__main__":
    main()
