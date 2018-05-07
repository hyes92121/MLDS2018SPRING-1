import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from vocabulary import Vocabulary

def checkpoint(): # currently not using this function
    print('program stopped at checkpoint. press Enter to continue')
    input()

class TrainingDataset(Dataset):
    def __init__(self, training_data_path, helper, load_into_ram=True): # TODO: currently loading all training text pairs into RAM
        # check if file path exists
        if not os.path.exists(training_data_path):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(training_data_path, __name__))


        self.training_data_path = training_data_path
        self.data_pair = [] # format: (prev_sentence[], curr_sentence[])
        self.load_into_ram = load_into_ram # whether to load all training text pairs into RAM
        self.helper = helper # this is a Vocabulary() class

        print('Parsing training data to Dataset()...')
        with open(self.training_data_path, 'r') as f:
            prev_sentence = []
            curr_sentence = []
            for idx, line in enumerate(f):
                line_words = line.split()
                if '+++$+++' in line_words:
                    prev_sentence = []
                    curr_sentence = []
                    # print('end of this conversation')
                else:
                    # shift sentences by 1
                    prev_sentence = curr_sentence
                    line_add_tokens = helper.reannotate(line)
                    curr_sentence = helper.sentence2index(line_add_tokens)
                    # if prev_sentence exists, add this training pair to data_pair
                    if prev_sentence != []:
                        self.data_pair.append((prev_sentence, curr_sentence))
        print('Finished creating Dataset() !')

    def __len__(self):
        return len(self.data_pair)


    def __getitem__(self, idx):
        """
        :returns: (_features, _sentence2index)
        """
        assert (idx < self.__len__())

        prev_sentence, curr_sentence = self.data_pair[idx]

        if not self.load_into_ram:
            print('please set load_into_ram to True!')
        else:
            return torch.Tensor(prev_sentence), torch.Tensor(curr_sentence)



def collate_fn(data): # data is a torch.Tensor batch returned by DataLoader()
    """
    Creates mini-batch tensors from the list of tuples (prev_sentence, curr_sentence).

        Args:
            data: list (batch_size) of tuple (prev_sentence, curr_sentence).
                - prev_sentence: torch tensor of shape (1, variable length).
                - curr_sentence: torch tensor of shape (1, variable length).
        Returns:
            padded_prev_sentences: torch tensor of shape (batch_size, pad_length_of_prev_sentences).
            lengths_prev_sentences: list; valid lengths (non-padded length)
            padded_curr_sentences: torch tensor of shape (batch_size, pad_length_of_curr_sentences).
            lengths_curr_sentences: list; valid lengths (non-padded length)
    """
    data.sort(key=lambda x: len(x[1]), reverse=True) # long ground truths first
    prev_sentences, curr_sentences = zip(*data)

    # pad input sentences
    lengths_prev_sentences = [len(sentence) for sentence in prev_sentences]
    padded_prev_sentences = torch.zeros(len(prev_sentences), max(lengths_prev_sentences)).long()
    for i, sentence in enumerate(prev_sentences):
        end = lengths_prev_sentences[i]
        padded_prev_sentences[i, -end:] = sentence[:end] # here we pad zeros at the BEGINNING of a sentence

    # pad ground truth sentences
    lengths_curr_sentences = [len(sentence) for sentence in curr_sentences]
    padded_curr_sentences = torch.zeros(len(curr_sentences), max(lengths_curr_sentences)).long()
    for i, sentence in enumerate(curr_sentences):
        end = lengths_curr_sentences[i]
        padded_curr_sentences[i, :end] = sentence[:end] # here we pad zeros at the END of a sentence
    
    return padded_prev_sentences, padded_curr_sentences, lengths_curr_sentences # only need lengths of ground truth (for calculating loss)



if __name__ == '__main__':
    import time
    from torch.autograd import Variable
#     from torch.nn.utils.rnn import pack_padded_sequence
#     from checkpoint import *

    training_data_path='data/clr_conversation.txt'

    helper = Vocabulary(training_data_path)

    dataset = TrainingDataset(training_data_path, helper, load_into_ram=True)

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=8, collate_fn=collate_fn)

    ss = time.time()

    for epoch in range(1):

        s = time.time()

        print('epoch: {}'.format(epoch+1))
        for batch_n, batch in enumerate(dataloader):

            #e = time.time()

            #print('batch No.{} time loading batch: {}'.format(batch_n, e-s))

            #s = time.time()
            print('batch no: {}'.format(batch_n))
            padded_prev_sentences, lengths_prev_sentences, padded_curr_sentences, lengths_curr_sentences = batch

            print(padded_prev_sentences)
            print()
            print(lengths_prev_sentences)
            print()
            print(padded_curr_sentences)
            print()
            print(lengths_curr_sentences)
            print()

#             checkpoint()

            for s in padded_prev_sentences:
                print(helper.index2sentence(s))
            print()
            
            for s in padded_curr_sentences:
                print(helper.index2sentence(s))
            print()

#             checkpoint()

            #packed = pack_padded_sequence(input=label, lengths=lengths, batch_first=True)
            #
            #print(packed.data)
            #
            #checkpoint()

            break
        e = time.time()

        #print('time for one epoch: {}'.format(e-s))

    ee = time.time()

    #print('total time: {}'.format(ee-ss))
