import time

import torch.nn as nn
import torch.optim as optim

import pickle
from nltk.translate import bleu_score
from sklearn.model_selection import train_test_split

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


def corpora2vectors(reverse_inputs=True, pad_spanish=True):
    with open(englishfile) as en_fd, open(spanishfile) as es_fd:
        garbage_filter = lambda x: x.strip() and x.strip() != '.'
        eng = [en_lang.tokens2tensor(en_lang.word_tokenize(s, reverse=reverse_inputs)) for s in en_fd]
        es = [es_lang.tokens2tensor(es_lang.word_tokenize(s, pad=pad_spanish)) for s in es_fd]
    return eng, es


en_lang, es_lang = build_full_vocabs()
X, y = corpora2vectors()
# with open('X.pkl', 'wb') as fd:
#     pickle.dump(X, fd)
# with open('y.pkl', 'wb') as fd:
#     pickle.dump(y, fd)

# filter vectors:
X, y = zip(*((i, j) for i, j in zip(X, y) if len(i) and len(j) != 1))
X = list(X)
y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

encoder = Encoder('encoder', vocab=en_lang, embedding_size=70, n_hidden=50, lstm_layers=2)
decoder = Decoder('decoder', vocab=es_lang, embedding_size=70, n_hidden=50, lstm_layers=2, local_window=2)
encoder = encoder.cuda()
decoder = decoder.cuda()

opt1 = optim.Adam(encoder.parameters(), lr=1e-4)
opt2 = optim.Adam(decoder.parameters(), lr=1e-4)

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

N = len(X_train)

for iter in range(1, 10000000):
    input_tensor = X_train[(iter - 1) % N].cuda()
    target_tensor = y_train[(iter - 1) % N].cuda()
    if iter % 20 == 0:
        loss = train(input_tensor, target_tensor, encoder, decoder, opt1, opt2, nn.NLLLoss(), step=True)
    else:
        loss = train(input_tensor, target_tensor, encoder, decoder, opt1, opt2, nn.NLLLoss(), step=False)
    print_loss_total += loss
    plot_loss_total += loss
    if iter % 1000 == 0:
        print_loss_avg = print_loss_total / 100
        print_loss_total = 0
        print('%.4f' % (print_loss_avg))
