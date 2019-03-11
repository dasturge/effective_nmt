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
        self.n_dims = n_hidden
        self.n_layers = lstm_layers

        # word embeddings:
        self.input_lookup = nn.Embedding(num_embeddings=vocab.size,
                                         embedding_dim=embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=self.n_dims,
                            num_layers=self.n_layers,
                            bidirectional=True)

    def forward(self, input, hidden):
        embedding = self.input_lookup(input)
        output = embedding
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(2 * self.n_layers, 1, self.n_dims)
        c0 = torch.zeros(2 * self.n_layers, 1, self.n_dims)
        return h0, c0


class Decoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int, local_window: int):
        super().__init__()
        self.__name__ = name

        n_pt_weights = n_hidden

        # Saving this so that other parts of the class can re-use it
        self.n_dims = n_hidden
        self.n_layers = lstm_layers
        self.local_window = local_window

        # word embeddings:
        self.output_lookup = nn.Embedding(num_embeddings=vocab.size,
                                          embedding_dim=embedding_size)

        # attention module
        self.p_t_dense = nn.Linear(n_hidden, n_pt_weights)
        self.p_t_dot = nn.Linear(n_pt_weights, 1)
        self.score = nn.Bilinear(self.n_dims, 2 * self.local_window + 1, 1)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=self.n_dims,
                            num_layers=self.n_layers,
                            bidirectional=False)

        self.dense_out = nn.Linear(self.n_dims, vocab.size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, h_t, h_s):
        embedding = self.output_lookup(input)

        # attention
        p_t = h_s.size(0) * F.sigmoid(self.p_t_dot(F.tanh(self.p_t_dense(h_t[0]))))  # (9)
        s = torch.round(p_t).long()
        D = self.local_window
        h_s_local = h_s[s - D:s + D + 1]  # @@@@@ zero pad! @@@@@
        score = self.score(h_t[0], h_s_local)  # (8) (general)  may have to use einsum here to do bilinear layer, or batch over source inputs
        gauss_window = torch.exp((torch.range(s - D, s + D + 1) - p_t) ** 2 / (D / 2) ** 2)  # (10)
        a_t = F.softmax(score) * gauss_window  # (7), (10)
        context = a_t * h_s_local
        hidden = torch.cat(context, h_t)

        # lstm
        output, h_t = self.lstm(embedding, hidden)
        output = F.log_softmax(self.dense_out(output[0]), dim=1)
        return output, h_t, (context, s)


def train(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn):
    encoder_hidden = encoder.init_hidden()  # is random any good here?

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = x.size(0)
    target_length = y.size(0)

    loss = 0
    x = x.view(-1, 1)
    h_s, h_t = encoder(x, encoder_hidden)

    bos = torch.tensor([[0]])  # BOS_TOKEN @@@@@@@ STARTING TO QUESTION THIS, SHOULDNT INPUT BE FINAL h_s??? @@@@@@@
    yhati = bos
    h_t = h_t  # ? for the beginning of the decoder I'm not sure how to handle this properly...  Ask Bedricks? Reinit?
    for di in range(target_length):
        yhati, h_t, decoder_attention = decoder(yhati, h_t, h_s)
        topv, topi = yhati.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += loss_fn(yhati, y[di])
        if decoder_input.item() == 1:  # EOS_TOKEN
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
