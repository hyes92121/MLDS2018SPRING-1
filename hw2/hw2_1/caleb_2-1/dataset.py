import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from vocabulary import Vocabulary



class TrainingDataset(Dataset):
    def __init__(self, label_json_file, training_data_path, helper, load_into_ram=False):
        # check if file path exists
        if not os.path.exists(label_json_file):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(label_json_file, __name__))
        if not os.path.exists(training_data_path):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(training_data_path, __name__))


        self.training_data_path = training_data_path
        self.data_pair = [] # format: (avi id, corresponding sentence2index)
        self.load_into_ram = load_into_ram # whether to load all .npy features into self.avi
        self.helper = helper # this is a Vocabulary() class


        with open(label_json_file, 'r') as f:
            label = json.load(f)

        for d in label:
            for s in d['caption']:
                s = self.helper.reannotate(s)
                s = self.helper.sentence2index(s)
                self.data_pair.append((d['id'], s))

        if load_into_ram:
            self.avi = {} # {_avi_id: _features}

            files = os.listdir(training_data_path)

            for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(training_data_path, file))
                self.avi[key] = value


    def __len__(self):
        return len(self.data_pair)


    def __getitem__(self, idx):
        """
        :returns: (_features, _sentence2index)
        """
        assert (idx < self.__len__())

        avi_file_name, sentence = self.data_pair[idx]

        if not self.load_into_ram:
            avi_file_path = os.path.join(self.training_data_path, '{}.npy'.format(avi_file_name))

            data = np.load(avi_file_path)

            return torch.Tensor(data), torch.Tensor(sentence)

        else:
            return torch.Tensor(self.avi[avi_file_name]), torch.Tensor(sentence)



def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).

        Args:
            data: list of tuple (avi_data, caption).
                - avi_data: torch tensor of shape (batch_size, 80, 4096).
                - caption: torch tensor of shape (batch_size, variable length).
        Returns:
            images: torch tensor of shape (batch_size, 80, 4096).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) # unzip data  format [(80, 4096), (80, 4096) ...]
                                    #             format [(1), (1)]

    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return avi_data, targets, lengths





if __name__ == '__main__':
    import time
    from torch.autograd import Variable
    from torch.nn.utils.rnn import pack_padded_sequence
    from checkpoint import *

    json_file = 'data/testing_label.json'
    numpy_file = 'data/testing_data/feat'

    helper = Vocabulary(json_file, min_word_count=5)

    dataset = TrainingDataset(label_json_file=json_file, training_data_path=numpy_file, helper=helper, load_into_ram=True)

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
            data, label, lengths = batch

            print(label[:, :12])
            print(lengths)

            checkpoint()

            for s in label:
                print(helper.index2sentence(s))

            checkpoint()

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





