import time

import torch.nn as nn
import torch.optim as optim

from model import Encoder, Decoder, train
from vocab import Vocab, VocabFull


# data data data
englishfile = 'data/europarl-v7.es-en_trunc.en'
spanishfile = 'data/europarl-v7.es-en_trunc.es'


def build_full_vocabs():
    with open(englishfile) as en_fd, open(spanishfile) as es_fd:
        en_lang = Vocab(name='english')
        es_lang = Vocab(name='spanish')
        try:
            en_lang.add_corpus(en_fd)
        except VocabFull:
            pass
        try:
            es_lang.add_corpus(es_fd)
        except:
            pass
    en_lang.calcify()
    es_lang.calcify()
    return en_lang, es_lang


def corpora2vectors():
    with open(englishfile) as en_fd, open(spanishfile) as es_fd:
        eng = [en_lang.tokens2tensor(en_lang.word_tokenize(s)) for s in en_fd]
        es = [es_lang.tokens2tensor(es_lang.word_tokenize(s)) for s in es_fd]
    return eng, es


en_lang, es_lang = build_full_vocabs()
X, y = corpora2vectors()

encoder = Encoder('encoder', vocab=en_lang, embedding_size=20, n_hidden=10, lstm_layers=1)
decoder = Decoder('decoder', vocab=en_lang, embedding_size=20, n_hidden=10, lstm_layers=1, local_window=3)

opt1 = optim.Adam(encoder.parameters())
opt2 = optim.Adam(decoder.parameters())

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

for iter in range(1, 10000):
    input_tensor = X[iter - 1]
    target_tensor = y[iter - 1]
    loss = train(input_tensor, target_tensor, encoder, decoder, opt1, opt2, nn.NLLLoss())
    print_loss_total += loss
    plot_loss_total += loss
    if iter % 100 == 0:
        print_loss_avg = print_loss_total / 100
        print_loss_total = 0
        print('%.4f' % (print_loss_avg))
