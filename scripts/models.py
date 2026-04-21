# Author: Matt Kuehr
# Email: mck0063@auburn.edu

import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    A recurrent neural network model for sentiment analysis, supporting RNN, GRU, and LSTM architectures.

    Attributes:
        embedding (nn.Embedding): The embedding layer to convert token IDs into dense vectors.
        rnn (nn.Module): The recurrent layer (RNN, GRU, or LSTM).
        fc (nn.Linear): The fully connected layer for binary classification.
        dropout (nn.Dropout): Dropout layer for regularization.
        sig (nn.Sigmoid): Sigmoid activation function for output probabilities.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, model_type="rnn"):
        """
        Initializes the SentimentRNN model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimensionality of the embedding vectors.
            hidden_dim (int): The dimensionality of the hidden state in the recurrent layer.
            output_dim (int): The number of output units.
            n_layers (int): The number of recurrent layers.
            bidirectional (bool): Whether to use a bidirectional recurrent layer.
            dropout (float): The dropout probability.
            model_type (str, optional): The type of recurrent layer to use ('rnn', 'gru', or 'lstm'). Defaults to "rnn".

        Raises:
            ValueError: If an invalid model_type is provided.
        """
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
        
    def forward(self, text, lengths):
        """
        Defines the forward pass of the model.

        Args:
            text (torch.Tensor): A batch of input token IDs, shape [batch_size, seq_len].
            lengths (torch.Tensor): A tensor containing the actual lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The output probabilities for each sequence, shape [batch_size, output_dim].
        """
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if isinstance(self.rnn, nn.LSTM):
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
            
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.sig(self.fc(hidden))
