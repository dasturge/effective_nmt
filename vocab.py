import string
import re
from collections import defaultdict

import nltk
import torch

import contractions  # see Vocab.sentence_preprocess docstring

# constants
BOS_SYM = '<B>'
EOS_SYM = '<E>'


class Vocab:

    def __init__(self, name, limit=None):
        # class for building a digital representation of a language.
        self.__name__ = name
        self.size = 0
        self.limit = None
        self.calcified = False
        self._token2index = defaultdict(self._next_index)
        self._index2token = None  # None until Vocab is calcified
        _ = self._token2index[BOS_SYM], self._token2index[EOS_SYM]

    @property
    def token2index(self):
        # return token2index dictionary
        return self._token2index

    @property
    def index2token(self):
        # return index2token dictionary
        if self.calcified:
            return self._index2token
        else:
            return {v: k for k, v in self.token2index.items()}

    def _next_index(self):
        # incrementer for building vocab
        if self.size == self.limit:
            self.calcify()
            raise VocabFull
        val = self.size
        self.size += 1
        return val

    def add(self, token):
        # add a token or list of tokens to the vocab
        if isinstance(token, list):
            _ = [self.add(t) for t in token]
        else:
            if self.size == self.limit:
                self.calcify()
                raise VocabFull
            _ = self._token2index[token]

    def add_sentence(self, sentence, language='english', verbose=False):
        # add an unprocessed sentence to the vocab
        if verbose:
            print(sentence)
        words = self.word_tokenize(sentence, language)
        self.add(words)

    def add_corpus(self, corpus, language='english'):
        # for even more brevity, add sentence tokenized corpus
        for sentence in corpus:
            sentence = sentence.strip()
            self.add_sentence(sentence, language=language)

    def calcify(self):
        # ensure no new tokens can be added, create unkown token
        items = self._token2index.items()
        unk_index = self.size
        self._token2index = defaultdict(lambda: unk_index, items)
        _ = self._token2index['<unk>']
        self._index2token = self.index2token
        self.calcified = True

    def tokens2vector(self, tokens, pad=True):
        # return a list of indices from a list of tokens
        t2i = self.token2index
        seq = [0] + [t2i[t] for t in tokens] + [1]
        return seq

    def tokens2tensor(self, tokens):
        # return a (1, n) tensor from list of tokens
        indices = self.tokens2vector(tokens)
        tensor = torch.Tensor(indices).long()
        return tensor

    @staticmethod
    def from_text(sentences, langid='noname'):
        voc = Vocab(name='noname')
        for sent in sentences:
            voc.add(sent)
        return voc

    @staticmethod
    def sent_tokenize(text, language='english'):
        # so you don't need to import nltk for just this
        return nltk.sent_tokenize(text, language=language)

    @staticmethod
    def word_tokenize(sentence, language='english', pad=False):
        # calls preprocessing on the sentence and then tokenizes
        sentence = Vocab.sentence_preprocess(sentence, language)
        words = nltk.word_tokenize(sentence, language=language)
        if pad:
            words = [BOS_SYM] + words + [EOS_SYM]
        return words

    @staticmethod
    def sentence_preprocess(s, language='english'):
        """
        reference for this nlp preprocessing:
        http://t-redactyl.io/blog/2017/06/text-cleaning-in-multiple-languages.html

        this just takes two methods from the blog post:
            - expanding common english contractions e.g. I'm, haven't (in turn cited from gist/stackoverflow)
            - removing punctuation, including special spanish: ¡¿ (code not copied here)
        """
        s = s.lower().strip()
        if language == 'english':
            s = contractions.expand_contractions(s)
        punct = re.compile('[' + string.punctuation + '¡¿' + ']')
        s = punct.sub('', s)
        return s


class VocabFull(Exception):
    pass
