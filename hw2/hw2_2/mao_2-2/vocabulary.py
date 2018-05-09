import os
import re
import json
from gensim.models import FastText
import numpy as np

class Vocabulary(object):# IMPORTANT: <embedding_filename>.npy is tied to the Vocabulary() that trained it
    """
    Helper class for processing words
    """

    def __init__(self, filepath, min_word_count=10, word_vec_dim=100, retrain_word_vecs=False, word_vec_outfile_name='word_vectors.npy'):

        # define public variables
        self.filepath = filepath
        self.min_word_count = min_word_count
        self.word_vec_dim = word_vec_dim


        # define private variables
        self._word_count = {}
        self.vocab_size = None # only counting good_words and useful_tokens
        self._good_words = None
        self._bad_words = None
        self.i2w = None # only counting good_words and useful_tokens
        self.w2i = None # only counting good_words and useful_tokens

        # initialize class
        print('Initalizing vocabulary...')
        self._initialize()

        print('Building mapping...')
        self._build_mapping()        
        print('Vocab size:', self.vocab_size)

        self._sanitycheck()

        if retrain_word_vecs:
            print('Retraining word vectors...')
            self._train_wordvecs(outfile_name=word_vec_outfile_name)


    def _initialize(self):
        """
        Runs through all words in training data, splitting them into
        bad words (low frequency) and good words
        :return:
        """
        with open(self.filepath) as f: # input file should be .txt
            for line in f:
                line_words = line.split()
                if '+++$+++' in line_words:
                    continue
                else:
                    for word in line_words: # not using re.sub(r'[.!,;?]', ' ', line)
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
        sentence = sentence.split()
        sentence = ['<SOS>'] + [w if (self._word_count.get(w, 0) > self.min_word_count) \
                                    else '<UNK>' for w in sentence] + ['<EOS>']

        return sentence

    def _train_wordvecs(self, outfile_name):
        with open(self.filepath) as f: # input file should be .txt
            model = FastText([self.reannotate(line) for line in f], min_count=self.min_word_count, size=self.word_vec_dim) # TODO: currently loading entire dataset into RAM
            
        word_vectors = np.zeros((len(self.i2w), self.word_vec_dim))
        for index in self.i2w: # let <embeddings_filename>.npy indices match Vocabulary() indices
            wordvec = model[self.i2w[index]]
            word_vectors[index] = wordvec
        print('Saving word vectors...')
        np.save(outfile_name, word_vectors)

    def word2index(self, w):
        return self.w2i[w]


    def index2word(self, i):
        return self.i2w[i]


    def sentence2index(self, sentence):
        return [self.w2i[w] for w in sentence]


    def index2sentence(self, index_seq):
        return [self.i2w[int(i)] for i in index_seq]

if __name__ == '__main__':
    retrain_word_vecs = False
    print('retrain_word_vecs:', retrain_word_vecs)
    a = Vocabulary('data/clr_conversation.txt', min_word_count=40, word_vec_dim=100, retrain_word_vecs = retrain_word_vecs)
    print(a.vocab_size) # (10,44089)  (20,29273)  (30,22410)  (40,19292)

    # print(len(a._good_words))
    # print(len(a._bad_words))

    # print(a._word_count['或類'])
    # print(a.reannotate("或類 軍事 組織 最愛 找 的 人"))

    # aa = a.reannotate("或類 軍事 組織 最愛 找 的 人")

    # bb = a.sentence2index(aa)
    # print(bb)

    # cc = a.index2sentence(bb)
    # print(cc)