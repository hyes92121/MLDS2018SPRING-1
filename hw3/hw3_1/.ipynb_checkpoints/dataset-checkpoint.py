import torch
import numpy as np
from torch.utils.data import Dataset
import os

class TrainingDataset(Dataset):
    """
    Training data
    """
    def __init__(self):
        print("start loading dataset...")
        if os.path.exists("data/extra_data.npy"):
            self.data = np.load("data/extra_data.npy")
        elif not os.path.exists("datasets/extra_data/images"):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format("datasets/extra_data/images", __name__))
        else:
            from glob import glob
            import skimage.io

            self.data = []
            pattern = os.path.join("datasets/extra_data/images", "*.jpg")   

            for _ , img in enumerate(glob(pattern)):
                self.data.append(skimage.io.imread(img))
                print('\r', "loading extra_dataset : ", _, "/ 36739", end='')

            self.data = np.array(self.data) #(_, 64, 64, 3)
            self.data = np.rollaxis(self.data, 3, 1) #(_, 3, 64, 64)
            self.data = (self.data-127.5)/127.5
            np.save("data/extra_data.npy", self.data)
            print()

        print("finished loading dataset...") # self.data[_] = (3, 64, 64)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        :returns: (image(64, 64, 3))
        """
        assert (idx < self.__len__())
        return torch.FloatTensor(self.data[idx])

if __name__ == '__main__':
    import time
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    dataset = TrainingDataset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=8)

    for epoch in range(1):
        for batch_n, batch in enumerate(dataloader):
            #print(batch)
            break