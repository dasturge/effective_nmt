import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab


class Encoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int):
        super().__init__()
        self.__name__ = name

        # Saving this so that other parts of the class can re-use it
        self.encoder_dims = n_hidden
        self.encoder_layers = lstm_layers

        # word embeddings:
        self.input_lookup = nn.Embedding(num_embeddings=vocab.size,
                                         embedding_dim=embedding_size)

        self.encoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=self.encoder_dims,
                               num_layers=self.encoder_layers,
                               bidirectional=True,
                               batch_first=True)

    def forward(self, input, hidden):
        embedding = self.input_lookup(input)
        output = embedding
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(self.encoder_layers, 1, self.encoder_dims)
        c0 = torch.zeros(self.encoder_layers, 1, self.encoder_dims)
        return h0, c0


class Decoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int):
        super().__init__()
        self.__name__ = name

        # Saving this so that other parts of the class can re-use it
        self.decoder_dims = n_hidden
        self.decoder_layers = lstm_layers

        # word embeddings:
        self.output_lookup = nn.Embedding(num_embeddings=vocab.size,
                                          embedding_dim=embedding_size)

        self.attention = lambda x, y: torch.einsum('', x, y)

        self.decoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=self.decoder_dims,
                               num_layers=self.decoder_layers,
                               bidirectional=True,
                               batch_first=True)

        self.dense = nn.Linear(self.decoder_dims, vocab.size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, h_t, h_s):
        embedding = self.output_lookup(input)
        output = F.relu(embedding)
        output, h_t = self.gru(output, h_t)
        output = self.dense(output[0])
        output = self.softmax(output)
        return output, h_t
