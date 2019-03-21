import pickle
import os
import re
import tarfile
import time

import requests
import torch
import torch.nn as nn
import torch.optim as optim

#from nltk.translate import bleu_score
from sklearn.model_selection import train_test_split

from model import Encoder, Decoder, NMT, VanillaDecoder
from vocab import Vocab, VocabFull


# data data data
englishfile = 'data/europarl-v7.es-en_truncated.en'
spanishfile = 'data/europarl-v7.es-en_truncated.es'

# collect our data
language = 'es'
tarfilename = "{}-en.tgz".format(language)
tarfilepath = os.path.join("data/", tarfilename)
def maybe_download():
    if not os.path.exists(tarfilepath):
        print('downloading {}...'.format(tarfilename))
        url = "http://www.statmt.org/europarl/v7/{}".format(tarfilename)
        os.makedirs('data/', exist_ok=True)
        r = requests.get(url, stream=True)
        with open(tarfilepath, 'wb') as fd:
            for content in r.iter_content():
                fd.write(content)
    if not os.path.exists(englishfile):
        print('download complete! Extracting...')
        with tarfile.open(tarfilepath) as tar:
            tar.extractall(path='data/')
        print('done!')
        
maybe_download()


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


def corpora2vectors(limit=None):
    with open(englishfile) as en_fd, open(spanishfile) as es_fd:
        eng = []
        esp = []
        reg = re.compile(r'[a-zA-Z]')
        for i, (en, es) in enumerate(zip(en_fd, es_fd)):
            en = en.strip()
            es = es.strip()
            if not reg.search(en):
                continue
            if not reg.search(es):
                continue
            eng.append(en_lang.tokens2tensor(en_lang.word_tokenize(en)))
            esp.append(es_lang.tokens2tensor(es_lang.word_tokenize(es)))
            if i == limit:
                break
    return eng, esp


if os.path.exists('en_lang.pkl'):
    with open('en_lang.pkl', 'rb') as fd:
        en_lang = pickle.load(fd)
    with open('es_lang.pkl', 'rb') as fd:
        es_lang = pickle.load(fd)
else:
    print('building vocabulary')
    en_lang, es_lang = build_full_vocabs()
    with open('en_lang.pkl', 'wb') as fd:
        try:
            pickle.dump(en_lang, fd)
        except Exception as e:
            import traceback
            traceback.print_exc()
    with open('es_lang.pkl', 'wb') as fd:
        try:
            pickle.dump(es_lang, fd)
        except Exception as e:
            import traceback
            traceback.print_exc()
if os.path.exists('X.pkl'):
    with open('X.pkl', 'rb') as fd:
        X = pickle.load(fd)
    with open('y.pkl', 'rb') as fd:
        y = pickle.load(fd)
else:
    print('building english, spanish data sets')
    X, y = corpora2vectors(limit=200000)
    # filter vectors:
    X, y = zip(*((i, j) for i, j in zip(X, y) if len(i) and len(j) != 1))
    X = list(X)
    y = list(y)
    with open('X.pkl', 'wb') as fd:
        pickle.dump(X, fd)
    with open('y.pkl', 'wb') as fd:
        pickle.dump(y, fd)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
with open('X_test.pkl', 'wb') as fd:
    pickle.dump(X_test, fd)
with open('y_test.pkl', 'wb') as fd:
    pickle.dump(y_test, fd)

encoder = Encoder('encoder', vocab=en_lang, embedding_size=150, n_hidden=100, lstm_layers=2)
decoder = Decoder('decoder', vocab=es_lang, embedding_size=150, n_hidden=100, lstm_layers=2, local_window=2)
encoder = encoder.cuda()
decoder = decoder.cuda()

encoder2 = Encoder('encoder', vocab=en_lang, embedding_size=150, n_hidden=100, lstm_layers=2)
decoder2 = VanillaDecoder('decoder', vocab=es_lang, embedding_size=150, n_hidden=100, lstm_layers=2)
encoder2 = encoder2.cuda()
decoder2 = decoder2.cuda()

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

batch_size = 10

# intialize counters
N = len(X_train)
epoch = 0
prev_percent = 0

nmt = NMT('nmt_local', encoder, decoder, nn.NLLLoss())
nmt_vanilla = NMT('nmt_vanilla', encoder2, decoder2, nn.NLLLoss())
import sys
sys.stdout = open('log.txt', 'a')
for tens in y_test[:5]:
    print(' '.join(es_lang.tensor2tokens(tens)))
for i in range(100):
    print('local')
    nmt.train(X_train, y_train, epochs=1, batch_size=1, print_every=10, examples=X_test[:5])
    nmt.save(path='nmt_local_%s' % i)
    print('no attention')
    nmt_vanilla.train(X_train, y_train, epochs=1, batch_size=1, print_every=10, examples=X_test[:5])
    nmt_vanilla.save(path='nmt_vanilla_%s' % i)

for i in range(10):
    nmt = NMT('nmt_')

if True:
    import sys
    sys.exit()
