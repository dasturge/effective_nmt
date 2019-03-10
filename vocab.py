from collections import defaultdict

import nltk
import torch

# constants
BOS_SYM = '<BOS>'
EOS_SYM = '<EOS>'


class Vocab:

    def __init__(self, name, pad=True):
        # class for building a digital representation of a language.
        self.__name__ = name
        self.pad = pad
        self.size = 0
        self._token2index = defaultdict(self._next_index)
        self._index2token = None
        _ = self._token2index[BOS_SYM], self._token2index[EOS_SYM]

    @property
    def token2index(self):
        # return token2index dictionary
        return self._token2index

    @property
    def index2token(self):
        # return index2token dictionary
        if self._index2token is not None:
            return self._index2token
        else:
            return {v: k for k, v in self.token2index.items()}

    def _next_index(self):
        # incrementer for building vocab
        val = self.size
        self.size += 1
        return val

    def add(self, token):
        # add a token or list of tokens to the vocab
        if isinstance(token, list):
            _ = [self.add(t) for t in token]
        else:
            _ = self._token2index[token]

    def calcify(self):
        # ensure no new tokens can be learned
        self._index2token = self.index2token
        self._token2index = dict(self._token2index)

    def tokens2vector(self, tokens):
        # return a list of indices from a list of tokens
        t2i = self.token2index
        if self.pad:
            tokens = [BOS_SYM] + tokens + [EOS_SYM]
        seq = [t2i[t] for t in tokens]
        return seq

    def tokens2tensor(self, tokens):
        # return a (1, n) tensor from list of tokens
        indices = self.tokens2vector(tokens)
        tensor = torch.Tensor(indices).long()
        return tensor

    @staticmethod
    def from_text(filename, pipeline=None):
        dispatcher = {
            'sentences': nltk.sent_tokenize,
            'words': nltk.word_tokenize,
        }
        pass
