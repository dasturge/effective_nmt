import torch
import torch.nn as nn
import torch.cuda

from vocab import Vocab

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Encoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int):
        super().__init__()
        self.__name__ = name

        # Saving this so that other parts of the class can re-use it
        self.n_hidden = n_hidden
        self.n_layers = lstm_layers

        # word embeddings:
        self.input_lookup = nn.Embedding(num_embeddings=vocab.size,
                                         embedding_dim=embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            bidirectional=False)

    def forward(self, input, hidden):
        embedding = self.input_lookup(input)
        output = embedding
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(self.n_layers, 1, self.n_hidden).cuda()
        c0 = torch.zeros(self.n_layers, 1, self.n_hidden).cuda()
        return h0, c0


class VanillaDecoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int, local_window: int):
        super().__init__()
        self.__name__ = name

        n_pt_weights = n_hidden

        # Saving this so that other parts of the class can re-use it
        self.n_hidden = n_hidden
        self.n_layers = lstm_layers
        self.local_window = local_window

        # word embeddings:
        self.output_lookup = nn.Embedding(num_embeddings=vocab.size,
                                          embedding_dim=embedding_size)

        # attention module
        self.p_t_dense = nn.Linear(self.n_hidden, n_pt_weights, bias=False)
        self.p_t_dot = nn.Linear(n_pt_weights, 1, bias=False)
        self.score = nn.Bilinear(self.n_hidden, self.n_hidden, 1, bias=False)  # ?

        self.combine_attention = nn.Linear(2 * self.n_hidden, self.n_hidden)

        self.lstm = nn.LSTM(input_size=embedding_size,  # + self.n_hidden
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            bidirectional=False,
                            dropout=0.2)

        self.dense_out = nn.Linear(self.n_hidden, vocab.size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, h_s, h_t_tilde):
        embedding = self.output_lookup(input.view(1, 1))
        # context_embedding = torch.cat((embedding, h_t_tilde), dim=-1)

        # lstm
        output, hidden = self.lstm(embedding, hidden)

        y = self.softmax(self.dense_out(output))
        a_t, p_t = None, None  # just to have the same inputs/outputs as the decoder
        return y, hidden, h_t_tilde, (a_t, p_t)


class Decoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int, local_window: int):
        super().__init__()
        self.__name__ = name

        n_pt_weights = n_hidden

        # Saving this so that other parts of the class can re-use it
        self.n_hidden = n_hidden
        self.n_layers = lstm_layers
        self.local_window = local_window

        # word embeddings:
        self.output_lookup = nn.Embedding(num_embeddings=vocab.size,
                                          embedding_dim=embedding_size)

        # attention module
        self.p_t_dense = nn.Linear(self.n_hidden, n_pt_weights, bias=False)
        self.p_t_dot = nn.Linear(n_pt_weights, 1, bias=False)
        self.score = nn.Bilinear(self.n_hidden, self.n_hidden, 1, bias=False)  # ?

        self.combine_attention = nn.Linear(2 * self.n_hidden, self.n_hidden)

        self.lstm = nn.LSTM(input_size=embedding_size + self.n_hidden,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            bidirectional=False,
                            dropout=0.2)

        self.dense_out = nn.Linear(self.n_hidden, vocab.size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, h_s, h_t_tilde):
        embedding = self.output_lookup(input.view(1, 1))

        context_embedding = torch.cat((embedding, h_t_tilde), dim=-1)

        # lstm
        output, hidden = self.lstm(context_embedding, hidden)

        # attention
        h_t = hidden[0][-1]  # ? correct?  Or should I concat on last axis?
        p_t = h_s.size(0) * torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(h_t)))).squeeze()  # (9)
        s = torch.round(p_t).long().cuda()
        D = self.local_window
        minimum, maximum = max(s - D, 0), min(s + D, h_s.size(0) - 1)
        h_s_local = h_s[minimum:maximum + 1]  # @@@@@ zero pad? @@@@@
        h_t_rep = h_t.repeat(h_s_local.size(0), 1, 1)
        score = self.score(h_t_rep, h_s_local)  # (8) (general)
        gauss_window = torch.exp((torch.arange(minimum, maximum + 1).float() - p_t) ** 2 / (D / 2) ** 2).view(-1, 1, 1).cuda()
        a_t = torch.softmax(score, dim=0) * gauss_window  # (7) & (10)
        context = torch.mean(a_t * h_s_local, dim=0, keepdim=True)
        h_t_tilde = torch.tanh(self.combine_attention(torch.cat((context, h_t.view(1, 1, -1)), dim=-1)))  # (5)
        y = self.softmax(self.dense_out(h_t_tilde))  # (6)

        return y, hidden, h_t_tilde, (a_t, p_t)


def train(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, step=True, clipping=0.25):
    hidden = encoder.init_hidden()

    target_length = y.size(0)

    loss = 0
    x = x.view(-1, 1)
    h_s, hidden = encoder(x, hidden)

    bos = torch.tensor([[0]]).cuda()  # BOS_TOKEN @@@@@@@ STARTING TO QUESTION THIS, SHOULDNT INPUT BE FINAL h_s??? @@@@@@@
    inp = bos
    h_t_tilde = hidden[0][-1].unsqueeze(0) * 0
    for di in range(target_length):
        logits, hidden, h_t_tilde, decoder_attention = decoder(inp, hidden, h_s, h_t_tilde)
        topv, topi = logits.topk(1)
        inp = topi.squeeze().detach()  # detach from history as input
        loss += loss_fn(logits.view(1, -1), y[di].view(1))
        if inp.item() == 1:  # EOS_TOKEN
            break

    loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), clipping)
    nn.utils.clip_grad_norm_(decoder.parameters(), clipping)

    if step:
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    return loss.item() / target_length
