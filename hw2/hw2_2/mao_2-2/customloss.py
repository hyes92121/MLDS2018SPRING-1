import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = None

    def forward(self, seq_Prob, curr_sentences_minus_SOS, lengths_curr_sentences):
        # seq_Prob: (batch, seq_len, vocab_size)
        # curr_sentences: (batch, seq_len)
        # lengths_curr_sentences: (batch, seq_len+1) because it has SOS
        try:
            assert(seq_Prob.shape[1] == curr_sentences_minus_SOS.shape[1])
        except AssertionError as error:
            print('Sequence lengths of seq_Prob and curr_sentences_minus_SOS (ground truth) do not match!')
       
        batch_size = len(seq_Prob)

        predict_cat = None # 2-dim Tensor of vocab softmaxes (total_len, vocab_size)
        groundT_cat = None # 1-dim Tensor of word indices (total_len,)

        flag = True

        for batch in range(batch_size):
            predict      = seq_Prob[batch]
            ground_truth = curr_sentences_minus_SOS[batch]
            length_curr_sentence_no_SOS = lengths_curr_sentences[batch]-1 # TODO: why do we not include <EOS> in loss calculation?

            # 
            predict = predict[:length_curr_sentence_no_SOS] # cut prediction length to be same as ground_truth length
            ground_truth = ground_truth[:length_curr_sentence_no_SOS]

            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False

            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)
                # concat so that we can calculate loss even if there exists
                # different sequence lengths in a batch

        try:
            assert len(predict_cat) == len(groundT_cat)

        except AssertionError as error:
            print('prediction length is not same as ground truth length. error location: Custom Loss')
            print('prediction length: {}, ground truth length: {}'.format(len(predict_cat), len(groundT_cat)))


        self.loss = self.loss_fn(predict_cat, groundT_cat)

        self.avg_loss = self.loss/batch_size # we need this, because CrossEntropyLoss() thinks batch size is 1 due to concat

        return self.loss


if __name__ == '__main__':
    from dataset import TrainingDataset, collate_fn
    from vocabulary import Vocabulary
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from model import VideoCaptionGenerator, EncoderRNN, DecoderRNN
    import time

    training_data_path='data/clr_conversation.txt'
    helper = Vocabulary(training_data_path)
    dataset = TrainingDataset(training_data_path, helper)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)
    
    encoder = EncoderRNN()
    decoder = DecoderRNN()
    model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

    for batch_idx, batch in enumerate(dataloader):
        padded_prev_sentences, padded_curr_sentences, lengths_curr_sentences = batch
        padded_prev_sentences, padded_curr_sentences = Variable(padded_prev_sentences), Variable(padded_curr_sentences)

        step = 50
        a = time.time()
        seq_Prob, seq_predictions = model(prev_sentences=padded_prev_sentences, mode='train', curr_sentences=padded_curr_sentences, steps=step)
        print('forward pass:', time.time()-a, 'seconds')

        loss_function = CustomLoss()

        a = time.time()
        loss = loss_function(seq_Prob, padded_curr_sentences[:, 1:], lengths_curr_sentences)
        print('calculate loss:', time.time()-a, 'seconds')

        a = time.time()
        loss.backward()
        print('backward pass:', time.time()-a, 'seconds')

        print(loss)
        break