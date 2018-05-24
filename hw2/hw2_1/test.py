import sys
import torch
from trainer import Trainer
from vocabulary import Vocabulary
from dataset import TestingDataset
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    model = torch.load('model/test_2.h5', map_location=lambda storage, loc: storage)
else:
    model = torch.load('model/test_2.h5')

dataset = TestingDataset('{}/feat'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

training_json_file = 'data/training_label.json'
helper = Vocabulary(training_json_file, min_word_count=3)

trainer = Trainer(model=model, test_dataloader=testing_loader, helper=helper)


for epoch in range(1):
    ss = trainer.test()

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))