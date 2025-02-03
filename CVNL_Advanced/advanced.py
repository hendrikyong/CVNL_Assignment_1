import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def text_to_sequence(text, vocab, max_len=10):
    tokens = [vocab.get(word, vocab["<unk>"]) for word in word_tokenize(text.lower())]
    return tokens[:max_len] + [vocab["<pad>"]] * (max_len - len(tokens))

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=True, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                            batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        if self.rnn.bidirectional:
            hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_dim)
            hidden_last = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            hidden_last = hidden[-1]
        hidden_last = self.dropout(hidden_last)
        output = self.fc(hidden_last)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.pickle", "rb") as f:
    vocab = pickle.load(f)
with open("intent_labels.pickle", "rb") as f:
    intent_labels = pickle.load(f)

idx_to_intent = {idx: intent for intent, idx in intent_labels.items()}
vocab_size = len(vocab)
embedding_dim = 50  
hidden_dim = 128  
output_dim = len(intent_labels)
num_layers = 1
bidirectional = True
dropout = 0.3

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim,
                      num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
model.load_state_dict(torch.load("CVNL_RNN.pth", map_location=device))
model.to(device)
model.eval()

def predict_intent(text):
    max_len = 10
    tokenized = text_to_sequence(text, vocab, max_len)
    input_tensor = torch.tensor(tokenized).unsqueeze(0).to(device)
    lengths = torch.tensor([min(len(word_tokenize(text.lower())), max_len)], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_tensor, lengths)
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
            print("Program closed")
            break
        
        intent = predict_intent(text)
        print("Predicted Intent:", intent, "\n")

if __name__ == "__main__":
    main()
