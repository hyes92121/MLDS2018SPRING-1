import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit

import logging
from MyLogger import logger, log_function

from checkpoint import *

logger = logger.getChild(__name__)



class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        logger.debug('Instance {} created'.format(__class__.__name__))
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        #self.match = nn.Linear(2 * hidden_size, hidden_size)
        self.match1 = nn.Linear(2*hidden_size, hidden_size)
        self.match2 = nn.Linear(hidden_size, hidden_size)
        self.match3 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)


    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- (1 x batch size x dim)
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """

        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)

        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.match1(matching_inputs)
        x = self.match2(x)
        x = self.match3(x)

        #attention_weights = self.to_weight(self.match(matching_inputs))
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_percentage=0.1):
        super(EncoderRNN, self).__init__()

        # hyper parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # layers
        self.compress = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_percentage)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        """
        param input: input sequence with shape (batch size, sequence_length)
        return: gru output, hidden state
        """
        batch_size, seq_len, feat_n = input.size()     # (-1, 80, 4096)
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.hidden_size) # compressed input features from 4096 to self.hidden_size

        output, hidden_state = self.gru(input)

        return output, hidden_state


class DecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size,
                 vocab_size,
                 word_dim,
                 helper=None,
                 dropout_percentage=0.1):
        super(DecoderRNN, self).__init__()

        # define hyper parameters
        self.hidden_size = hidden_size # size of gru's Y and H
        self.output_size = output_size # final decoder output dim
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.helper = helper


        # define layers
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', steps=None):
        """
        :param encode_hidden_state:
        :param encoder_output:
        :param targets: (batch, seq_len) target labels of ground truth sentences
        :return:
        """

        # parameters used in both train and inference stage
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  #<SOS> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []


        if targets is None:
            raise NotImplementedError('Training target is None. Error location: RNNDecoder')
        if steps is None:
            raise NotImplementedError('steps is not specified. Error location: RNNDecoder -> steps')

        #TODO: implement schedule sampling

        targets = self.embedding(targets) # (batch, max_seq_len, embedding_size) embeddings of target labels of ground truth sentences
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1): # only the MAX_SEQ_LEN-1 words will be the gru input
            """
            we implement the decoding procedure in a step by step fashion
            so the seq_len is always 1
            """
            threshold = self._get_teacher_learning_ratio(training_steps=steps)

            current_input_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold \
                else self.embedding(decoder_current_input_word).squeeze(1)
            
            #current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            #if random.uniform(0.05, 0.995) > threshold:
            #    current_input_word = targets[:, i]
            #    print('using teacher word')
            #else:
            #    current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            #    print('using self word')

            logger.debug('encoder output size: {}'.format(encoder_output.size()))
            logger.debug('decoder current hidden state: {}'.format(decoder_current_hidden_state.size()))


            # weighted sum of the encoder output w.r.t the current hidden state
            context = self.attention(decoder_current_hidden_state, encoder_output)

            logger.debug('current input word size: {}'.format(current_input_word.size()))
            logger.debug('context size: {}'.format(context.size()))

            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)

            logger.debug('gru input size: {}'.format(gru_input.size()))
            logger.debug('decoder current hidden state size: {}'.format(decoder_current_hidden_state.size()))

            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)

            logger.debug('gru output size: {}'.format(gru_output.size()))
            logger.debug('decoder output hidden state size: {}'.format(decoder_current_hidden_state.size()))

            # project the dim of the gru output to match the final decoder output dim
            #logprob = F.log_softmax(self.to_final_output(gru_output.squeeze(1)), dim=1)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))

            logger.debug('size of every output stage: {}'.format(logprob.size()))

            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        # after calculating all word prob, concatenate seq_logProb into dim(batch, seq_len, output_size)
        seq_logProb = torch.cat(seq_logProb, dim=1)

        logger.debug('total output seq size: {}'.format(seq_logProb.size()))

        seq_predictions = seq_logProb.max(2)[1]

        return seq_logProb, seq_predictions


    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  # <SOS> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []

        assumption_seq_len = 28
        for i in range(assumption_seq_len-1):

            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            logger.debug('encoder output size: {}'.format(encoder_output.size()))
            logger.debug('decoder current hidden state: {}'.format(decoder_current_hidden_state.size()))

            context = self.attention(decoder_current_hidden_state, encoder_output)

            logger.debug('current input word size: {}'.format(current_input_word.size()))
            logger.debug('context size: {}'.format(context.size()))

            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)

            logger.debug('gru input size: {}'.format(gru_input.size()))
            logger.debug('decoder current hidden state size: {}'.format(decoder_current_hidden_state.size()))

            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)

            logger.debug('gru output size: {}'.format(gru_output.size()))
            logger.debug('decoder output hidden state size: {}'.format(decoder_current_hidden_state.size()))

            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))

            logger.debug('size of every output stage: {}'.format(logprob.size()))

            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)

        logger.debug('total output seq size: {}'.format(seq_logProb.size()))

        seq_predictions = seq_logProb.max(2)[1]

        return seq_logProb, seq_predictions


    def initialize_hidden_state(self, last_encoder_hidden_state):
        if last_encoder_hidden_state is None:
            return None

        else:
            return last_encoder_hidden_state


    def _get_teacher_learning_ratio(self, training_steps):
        return (expit(training_steps/40 +0.85))



class VideoCaptionGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super(VideoCaptionGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, avi_feats, mode, target_sentences=None, steps=None):
        """
        Args:
            param avi_feats(Variable): size(batch size x 80 x 4096)
            param target_sentences: ground truth for training, None for inference
        Returns:
            seq_logProb
            seq_predictions
        """

        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feats)

        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(
                encoder_last_hidden_state = encoder_last_hidden_state,
                encoder_output = encoder_outputs,
                targets = target_sentences,
                mode = mode,
                steps=steps
            )

        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_output=encoder_outputs,
            )

        else:
            raise KeyError('mode is not valid')

        return seq_logProb, seq_predictions




if __name__ == '__main__':
    import logging
    logger.setLevel(logging.INFO)
    from vocabulary import Vocabulary

    json_file = 'data/testing_label.json'
    numpy_file = 'data/testing_data/feat'

    helper = Vocabulary(json_file, min_word_count=5)



    input_data = Variable(torch.randn(3, 80, 4096).view(-1, 80, 4096))

    encoder = EncoderRNN(input_size=4096, hidden_size=1000)
    decoder = DecoderRNN(hidden_size=1000, output_size=1700, vocab_size=1700, word_dim=128, helper=helper)

    model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

    ground_truth = Variable(torch.rand(3, 27)).long()

    for step in range(50, 100):
        seq_prob, seq_predict = model(avi_feats=input_data, mode='train', target_sentences=ground_truth, steps=step)

        if step % 10 == 0:
            print(seq_prob.size())
            print(seq_predict.size())



















