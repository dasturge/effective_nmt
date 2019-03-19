import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda

from vocab import Vocab

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Encoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int):
        """
        Basic Encoder Module for neural machine translation
        :param name: module name
        :param vocab: Vocab object, for mapping numerics to words
        :param embedding_size: size of word embedding
        :param n_hidden: number of hidden nodes in each lstm.
        :param lstm_layers: number of lstm layers
        """
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
        """
        basic forward pass of encoder, defined for full inputs
        :param input: encoder lstm inputs
        :param hidden: previous lstm hidden state
        :return:
        """
        embedding = self.input_lookup(input)
        output = embedding
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(self.n_layers, 1, self.n_hidden).cuda()
        c0 = torch.zeros(self.n_layers, 1, self.n_hidden).cuda()
        return h0, c0


class Decoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int, local_window: int):
        """
        Decoder Module with Attention Mechanism for neural machine translation
        :param name: name for object instance
        :param vocab: output vocabulary for predictions
        :param embedding_size:
        :param n_hidden:
        :param lstm_layers:
        :param local_window:
        """
        super().__init__()
        self.__name__ = name

        n_pt_weights = n_hidden

        # Saving this so that other parts of the class can re-use it
        self.n_hidden = n_hidden
        self.n_layers = lstm_layers
        self.out_lut = vocab.tensor2tokens
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
        if self.local_window:
            # local
            h_t = hidden[0][-1]
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
        else:
            # global
            pass

        h_t_tilde = torch.tanh(self.combine_attention(torch.cat((context, h_t.view(1, 1, -1)), dim=-1)))  # (5)
        y = self.softmax(self.dense_out(h_t_tilde))  # (6)

        return y, hidden, h_t_tilde, (a_t, s)


class NMT:

    def __init__(self, name, encoder, decoder, loss_fn):
        """
        simple wrapper class for handling encoder-decoder structure.
        :param name: Name for NMT instance
        :param encoder: Encoder module instance
        :param decoder: Decoder module instance (must match hidden state size of encoder)
        :param loss_fn: a loss function for comparison with log-softmax layer. Probably NLLLoss(dim=-1)
        """
        self.__name__ = name
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam((*encoder.parameters(), *decoder.parameters()), lr=1e-3)
        self.epochs = 0

    def set_optimizer(self, optimizer, learning_rate=1e-3):
        """
        explicitly set the optimizer
        :param optimizer: new optimizer
        :param learning_rate: new learning rate
        :return:
        """
        self.optimizer = optimizer((*self.encoder.parameters(),
                                    *self.decoder.parameters()),
                                   lr=learning_rate)

    def train(self, X, y, epochs=1, batch_size=1, clipping=0.25, print_every=4, examples=[],
              examples_epoch_fn=lambda x: True):
        """
        epoch training function
        :param X: input training data
        :param y: output training data
        :param epochs: number of epochs to train
        :param batch_size: batch size (WARNING: Empirically doesn't work)
        :param clipping: gradient clipping coef
        :param print_every: print loss at every N% completion of each epoch
        :param examples: optional training examples for displaying progress.
        :param examples_epoch_fn: function which takes the epoch number, and returns a boolean value.
        Training examples are only printed for epochs where the function returns true.
        e.g. lambda x: x**(1/2) == int(x**(1/2))
        :return: None
        """
        N = len(X)

        for e in range(epochs):
            print_loss_total = prev_log_iter = prev_log_percent = 0
            for n in range(N):
                xi = X[n]
                yi = y[n]
                step = not n % batch_size or n == N - 1
                print_loss_total += self._train(xi, yi, clipping, step=step, loss_factor=batch_size) * batch_size
                if step and 100 * n // N > prev_log_percent + print_every:
                    print_loss_avg = print_loss_total / (n - prev_log_iter)
                    prev_log_iter = n
                    prev_log_percent = 100 * n // N
                    print('\repoch %s: %%%s complete     avg loss: %.4f' %
                          (self.epochs, prev_log_percent, print_loss_avg))
                    print_loss_total = 0
            self.epochs += 1
            if examples_epoch_fn(e):
                for ex in examples:
                    output = self.predict(ex)
                    print(' '.join(self.decoder.out_lut(output)))

    def predict(self, input, cap: int = 20):
        """
        returns prediction vector from input
        :param input: input statement
        :param cap: max length for prediction.
        :return:
        """
        hidden = self.encoder.init_hidden()
        encoded, hidden = self.encoder(input.view(-1, 1), hidden)
        inp = torch.tensor([[1]]).cuda()  # EOS_TOKEN
        h_t_tilde = hidden[0][-1].unsqueeze(0) * 0
        output = []
        for di in range(cap):
            logits, hidden, h_t_tilde, decoder_attention = self.decoder(inp, hidden, encoded, h_t_tilde)
            topv, topi = logits.topk(1)
            inp = topi.squeeze().detach()  # detach from history as input
            if inp.item() == 1:  # EOS_TOKEN
                break
            output.append(inp.squeeze())

        return torch.stack(tuple(output), dim=0)

    def save(self, path=None):
        if not path:
            path = os.path.join('.', self.name)
        encoder = os.path.join(path, '_encoder')
        decoder = os.path.join(path, '_decoder')
        torch.save(self.encoder.state_dict(), encoder)
        torch.save(self.decoder.state_dict(), decoder)

    @staticmethod
    def load(path, name='nmt'):
        encoder = os.path.join(path, '_encoder')
        decoder = os.path.join(path, '_decoder')
        enc = torch.load(encoder)
        dec = torch.load(decoder)
        encoder = Encoder.load_state_dict(enc)
        decoder = Decoder.load_state_dict(dec)
        nmt = NMT(name, encoder, decoder, loss_fn=nn.NLLLoss())
        return nmt

    def _train(self, x, y, clipping, step, loss_factor=1):
        """
        single example training
        :param x: single input sentence vector
        :param y: single output sentence vector
        :param clipping: gradient clipping
        :param step: execute optimizer step or not
        :param loss_factor: basic factor for batch averaging.
        :return:
        """
        hidden = self.encoder.init_hidden()

        target_length = y.size(0)

        loss = 0
        x = x.view(-1, 1)
        h_s, hidden = self.encoder(x, hidden)

        bos = torch.tensor(
            [[0]]).cuda()  # BOS_TOKEN
        inp = bos
        h_t_tilde = hidden[0][-1].unsqueeze(0) * 0
        for di in range(target_length):
            logits, hidden, h_t_tilde, decoder_attention = self.decoder(inp, hidden, h_s, h_t_tilde)
            topv, topi = logits.topk(1)
            inp = topi.squeeze().detach()  # detach from history as input
            loss += self.loss_fn(logits.view(1, -1), y[di].view(1)) / loss_factor
            if inp.item() == 1:  # EOS_TOKEN
                break
            inp = y[di]

        loss.backward()
        nn.utils.clip_grad_norm_((*self.encoder.parameters(), *self.decoder.parameters()), clipping)

        if step:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item() / target_length


class VanillaDecoder(nn.Module):

    def __init__(self, name: str, vocab: Vocab, embedding_size: int,
                 n_hidden: int, lstm_layers: int, local_window: int):
        """
        Decoder Module without an attention mechanism, for comparison.
        See Decoder for parameter help.
        """
        super().__init__()
        self.__name__ = name

        n_pt_weights = n_hidden
        self.lut = vocab.tokens2tensor

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
        return y, hidden, h_t_tilde, \
            (None, None)  # for conformed interfacing
