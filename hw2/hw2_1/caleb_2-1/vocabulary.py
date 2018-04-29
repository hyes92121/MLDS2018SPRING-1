import os
import re
import json


class Vocabulary(object):
    """
    Helper class for processing words
    """

    def __init__(self, filepath, min_word_count=10):

        # define public variables
        self.filepath = filepath
        self.min_word_count = min_word_count


        # define private variables
        self._word_count = {}
        self.vocab_size = None # only counting good_words and useful_tokens
        self._good_words = None
        self._bad_words = None
        self.i2w = None # only counting good_words and useful_tokens
        self.w2i = None # only counting good_words and useful_tokens

        # initialize class
        self._initialize()
        self._build_mapping()
        self._sanitycheck()


    def _initialize(self):
        """
        Runs through all words in training data, splitting them into
        bad words (low frequency) and good words
        :return:
        """
        with open(self.filepath, 'r') as f:
            file = json.load(f)

        for d in file:
            for s in d['caption']:
                word_sentence = re.sub('[.!,;?]]', ' ', s).split()

                for word in word_sentence:
                    word = word.replace('.', '') if '.' in word else word
                    self._word_count[word] = self._word_count.get(word, 0) + 1


        bad_words = [k for k, v in self._word_count.items() if v <= self.min_word_count]
        vocab = [k for k, v in self._word_count.items() if v > self.min_word_count]

        self._bad_words = bad_words
        self._good_words = vocab


    def _build_mapping(self):
        """
        build dictionaries for mapping word to index and vice versa
        :return: None
        """
        useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]

        self.i2w = {i + len(useful_tokens): w for i, w in enumerate(self._good_words)}
        self.w2i = {w: i + len(useful_tokens) for i, w in enumerate(self._good_words)}

        for token, index in useful_tokens:
            self.i2w[index] = token
            self.w2i[token] = index

        self.vocab_size = len(self.i2w)


    def _sanitycheck(self):
        attrs = ['vocab_size', '_good_words', '_bad_words', 'i2w', 'w2i']

        for att in attrs:
            if getattr(self, att) is None:
                raise NotImplementedError('Class {} has an attribute "{}" which cannot be None. Error location: {}'.format(__class__.__name__, att, __name__))


    def reannotate(self, sentence):
        """
        replaces word with <UNK> if word is infrequent
        :param sentence:
        :return: reannotated sentence
        """
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = ['<SOS>'] + [w if (self._word_count.get(w, 0) > self.min_word_count) \
                                    else '<UNK>' for w in sentence] + ['<EOS>']

        return sentence


    def word2index(self, w):
        return self.w2i[w]


    def index2word(self, i):
        return self.i2w[i]


    def sentence2index(self, sentence):
        return [self.w2i[w] for w in sentence]


    def index2sentence(self, index_seq):
        return [self.i2w[int(i)] for i in index_seq]


if __name__ == '__main__':

    a = Vocabulary('data/training_label.json', 2)
    print(a.vocab_size)
    print(len(a._good_words))
    print(len(a._bad_words))
    print(a._word_count['pen'])
    print(a.reannotate("A chicken is being seasoned."))

    aa = a.reannotate("A chicken is being seasoned.")

    bb = a.sentence2index(aa)
    print(bb)

    cc = a.index2sentence(bb)
    print(cc)

















