import time

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
from nltk.translate import bleu_score
from sklearn.model_selection import train_test_split

from model import Encoder, Decoder, NMT
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

encoder = Encoder('encoder', vocab=en_lang, embedding_size=70, n_hidden=70, lstm_layers=2)
decoder = Decoder('decoder', vocab=es_lang, embedding_size=70, n_hidden=70, lstm_layers=2, local_window=2)
encoder = encoder.cuda()
decoder = decoder.cuda()

opt1 = optim.Adam(encoder.parameters(), lr=1e-4)
opt2 = optim.Adam(decoder.parameters(), lr=1e-4)

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

batch_size = 10

# intialize counters
N = len(X_train)
epoch = 0
prev_percent = 0

# cudafy inputs
for i, (xi, yi) in enumerate(zip(X_train, y_train)):
    X_train[i] = xi.cuda()
    y_train[i] = yi.cuda()


nmt = NMT(encoder, decoder, nn.NLLLoss())
nmt.train(X_train, y_train, epochs=100, batch_size=1, examples=X_test[:5])

if True:
    import sys
    sys.exit()
for iter in range(1, 10000000):
    input_tensor = X_train[(iter - 1) % N]
    target_tensor = y_train[(iter - 1) % N]
    if iter % batch_size == 0:
        loss = train(input_tensor, target_tensor, encoder, decoder, opt1, opt2, nn.NLLLoss(), step=True)
    else:
        loss = train(input_tensor, target_tensor, encoder, decoder, opt1, opt2, nn.NLLLoss(), step=False)
    print_loss_total += loss
    plot_loss_total += loss
    if (iter - (N // iter)) / 100 > prev_percent:
        prev_percent += 1
        print_loss_avg = print_loss_total / 1000
        print_loss_total = 0
        print('\repoch %s: %%%s  avg loss: %.4f' % (epoch, prev_percent, print_loss_avg))
    if iter % N == 0:
        epoch += 1
        prev_percent = 0
        if epoch >= 12:
            for param in opt1.param_groups:
                param['lr'] /= 2
            for param in opt2.param_groups:
                param['lr'] /= 2
        encoded, hidden = encoder(X_test[5].view(-1, 1), encoder.init_hidden())
        inp = torch.tensor([[1]]).cuda()  # EOS_TOKEN
        h_t_tilde = hidden[0][-1].unsqueeze(0) * 0
        output = []
        for di in range(20):
            logits, hidden, h_t_tilde, decoder_attention = decoder(inp, hidden, encoded, h_t_tilde)
            topv, topi = logits.topk(1)
            inp = topi.squeeze().detach()  # detach from history as input
            if inp.item() == 1:  # EOS_TOKEN
                break
            output.append(inp)
        print(' '.join(decoder.out_lut(output)))
