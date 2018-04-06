import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from MyLogger import logger
from MyLogger import log_method, log_function

logger = logger.getChild(__name__)


class TrainingDataset(Dataset):
    logger.debug("Creating {} instance".format(__name__))
    def __init__(self, csvpath):
        self.data       = pd.read_csv(csvpath)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        assert idx < self.__len__()
        x, y = self.data.iloc[idx]

        return torch.Tensor([x, y])

    # implement if necessary
    def parse_data(self):
        pass


if __name__ == "__main__":     # Testing code making sure that this class is working

    csvpath = "data_train.csv"
    rootdir = "HW1/"

    dataset = TrainingDataset(csvpath=csvpath, rootdir=rootdir)

    batch_size  = 20
    shuffle     = True
    num_workers = 1


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for batch_n, batch in enumerate(dataloader):
        data, target = batch.t()

        if batch_n%5 == 0:
            print(batch_n, data, target)

        break










