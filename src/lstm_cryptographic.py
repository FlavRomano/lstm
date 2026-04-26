from torch import nn

class LSTMCryptographic(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMCryptographic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
            x = self.embedding(x) 
            output, (hn, cn) = self.lstm(x)
            
            return self.fc(hn[-1])