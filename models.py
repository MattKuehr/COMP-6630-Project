import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, model_type="rnn"):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if model_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=n_layers, 
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, 
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, 
                              bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            raise ValueError("Invalid model_type. Choose 'rnn', 'gru', or 'lstm'.")
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        
    def forward(self, text):
        # text: [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch size, sent len, emb dim]
        
        if isinstance(self.rnn, nn.LSTM):
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)
            
        # hidden: [num layers * num directions, batch size, hid dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # hidden: [batch size, hid dim * num directions]
        return self.sig(self.fc(hidden))
