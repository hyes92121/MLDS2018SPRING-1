import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.match = nn.Linear(2*hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)


    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            (decoder current) hidden_state {Variable} -- (1, batch, hidden_size)
            encoder_outputs {Variable} -- (batch, seq_len, hidden_size) 
        Returns:
            Variable -- context vector of size batch_size x dim
        """

        batch_size, seq_len, feat_n = encoder_outputs.size()
        # Resize hidden_state and copy it seq_len times, so that we can get its attention
        # with each encoder_output
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)

        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        attention_weights = self.to_weight(self.match(matching_inputs))
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context


class EncoderRNN(nn.Module):
    def __init__(self, word_vec_filepath='word_vectors.npy', hidden_size=1024, num_layers=2): # TODO: use more than 1 layer (may have to adjust other modules)
        super(EncoderRNN, self).__init__()
    
        self.hidden_size = hidden_size
        
        # load pretrained embedding
        pretrained = np.load(word_vec_filepath)
        self.vocab_size = pretrained.shape[0]
        self.word_vec_dim = pretrained.shape[1]
        
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.word_vec_dim)
        embedding.weight = nn.Parameter(torch.Tensor(pretrained)) # requires_grad == True
        self.embedding = embedding # TODO: can let encoder and decoder share embeddings
        
        # feed word vector into encoder GRU
        self.gru = nn.GRU(input_size=self.word_vec_dim, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input): # input: (batch_size, sentence_length)
        """
        param input: input sequence with shape (batch size, sequence_length)
        return: gru output, hidden state
        """    
        word_embeddings = self.embedding(input) # (batch_size, sentence_length, word_vec_dim)
        top_layer_output, last_time_step_all_layers_output = self.gru(word_embeddings)
        # top_layer_output: (seq_len, batch, hidden_size * num_directions)
        # last_time_step_all_layers_output: (num_layers * num_directions, batch, hidden_size)
        
        return top_layer_output, last_time_step_all_layers_output

class DecoderRNN(nn.Module):
    def __init__(self, word_vec_filepath='word_vectors.npy', hidden_size=1024, num_layers=2): # TODO: use more than 1 layer (may have to adjust other modules)
        super(DecoderRNN, self).__init__()

        # define hyper parameters
        self.hidden_size = hidden_size # size of gru's Y and H
        
        # load pretrained embedding
        pretrained = np.load(word_vec_filepath)
        self.vocab_size = pretrained.shape[0]
        self.word_vec_dim = pretrained.shape[1]
        
        embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.word_vec_dim)
        embedding.weight = nn.Parameter(torch.Tensor(pretrained)) # requires_grad == True
        self.embedding = embedding # TODO: can let encoder and decoder share embeddings

        # gru input is word vector of prev_output_word (one hot), plus attention context vector
        self.gru = nn.GRU(self.word_vec_dim+self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = AttentionLayer(self.hidden_size)
        # output is softmax over entire vocabulary
        self.to_final_output = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', steps=None):
        """
        :param encoder_last_hidden_state: (num_layers * num_directions, batch, hidden_size)
        :param encoder_output: (batch, length_prev_sentences, hidden_size * num_directions)
        :param targets: (batch, length_curr_sentences) target ground truth sentences
        :param steps: just a parameter used for calculating scheduled sampling, unrelated to RNN time steps
        :return:
        """

        # parameters used in both train and inference stage
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = encoder_last_hidden_state # (encoder_num_layers * num_directions, batch, hidden_size)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  #<SOS> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_Prob = []
        seq_predictions = []


        if targets is None:
            raise NotImplementedError('Training target is None. Error location: RNNDecoder')
        if steps is None:
            raise NotImplementedError('steps is not specified. Error location: RNNDecoder -> steps')

        # targets is only used for scheduled sampling, not used for calculating loss
        targets = self.embedding(targets) # (batch, max_seq_len, embedding_size) embeddings of target labels of ground truth sentences
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1): # Decoder will never have EOS as input
            """
            we implement the decoding procedure in a step by step fashion
            so the seq_len is always 1
            """
            threshold = self._get_teacher_learning_ratio(training_steps=steps)
            
            # target[:, i]: (batch, 1, embedding_size)
            current_input_word = targets[:, i] if random.random() < threshold \
                else self.embedding(decoder_current_input_word)
            # current_input_word: (batch, 1, embedding_size)

            # weighted sum of the encoder output w.r.t the current hidden state
            context = self.attention(decoder_current_hidden_state[-1], encoder_output) # (1, batch, hidden_size) (batch, seq_len, hidden_size) 
            # context: (batch, hidden_size)
            gru_input = torch.cat([current_input_word.squeeze(1), context], dim=1).unsqueeze(1)
            # gru_input: (batch, 1, embedding_size+hidden_size)

            # only runs for one time step because sequence length is only 1
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            # gru_output (last time step): (batch, seq_length==1, hidden_size * num_directions)
            # decoder_current_hidden_state (last layer): (num_layers * num_directions, batch, hidden_size)

            # project the dim of the gru output to match the final decoder output dim
            # logprob = F.log_softmax(self.to_final_output(gru_output.squeeze(1)), dim=1)
            prob = self.to_final_output(gru_output.squeeze(1)) # prob: (batch, vocab_size)
            seq_Prob.append(prob)

            decoder_current_input_word = prob.max(1)[1]
            
        # seq_Prob: list of [(batch, vocab_size), (batch, vocab_size)], len(list) == seq_len
        seq_Prob = torch.stack(seq_Prob, dim=1)
        # seq_Prob: (batch, seq_len, vocab_size)
        
        seq_predictions = seq_Prob.max(2)[1]
        # seq_predictions: (batch, seq_length)

        return seq_Prob, seq_predictions

    # basically same as forward(), but without scheduled sampling
    def infer(self, encoder_last_hidden_state, encoder_output, assumption_seq_len=28):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = encoder_last_hidden_state # (encoder_num_layers * num_directions, batch, hidden_size)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  #<SOS> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_Prob = []
        seq_predictions = []

        for i in range(assumption_seq_len-1): # run for fixed amount of time steps

            current_input_word = self.embedding(decoder_current_input_word)

            context = self.attention(decoder_current_hidden_state[-1], encoder_output)

            gru_input = torch.cat([current_input_word.squeeze(1), context], dim=1).unsqueeze(1)

            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)

            prob = self.to_final_output(gru_output.squeeze(1))
            seq_Prob.append(prob)

            decoder_current_input_word = prob.max(1)[1]

        seq_Prob = torch.stack(seq_Prob, dim=1)

        seq_predictions = seq_Prob.max(2)[1]

        return seq_Prob, seq_predictions
    
    # beammmmmm search!
    def beam_search(self, encoder_last_hidden_state, encoder_output, assumption_seq_len=27):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = encoder_last_hidden_state # (encoder_num_layers * num_directions, batch, hidden_size)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  #<SOS> (batch x word index)
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        #seq_Prob_1 = 0
        #seq_Prob_2 = 0
        seq_predictions_1 = []
        seq_predictions_2 = []
        
        k = 2
        
        current_input_word = self.embedding(decoder_current_input_word)
        context = self.attention(decoder_current_hidden_state[-1], encoder_output)
        gru_input = torch.cat([current_input_word.squeeze(1), context], dim=1).unsqueeze(1)
        gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
        prob = self.to_final_output(gru_output.squeeze(1))
        logprob = F.log_softmax(prob, dim=1)
        beam_word = logprob.topk(k, dim=1, largest=True, sorted=True)
        
            
        decoder_current_input_word_1 = beam_word[1][0][0].clone()
        decoder_current_input_word_2 = beam_word[1][0][1].clone()
        decoder_current_hidden_state_1 = decoder_current_hidden_state.clone()
        decoder_current_hidden_state_2 = decoder_current_hidden_state.clone()
        
        seq_predictions_1 = (decoder_current_input_word_1)
        seq_predictions_2 = (decoder_current_input_word_2)
        
        #print(seq_predictions_1)
        #print(seq_predictions_2)
        
        seq_Prob_1 = beam_word[0][0][0].clone()
        seq_Prob_2 = beam_word[0][0][1].clone()
        #print(seq_Prob_1)
        #print(seq_Prob_2)
        
        
        for i in range(assumption_seq_len-1): # run for fixed amount of time steps

            current_input_word_1 = self.embedding(decoder_current_input_word_1)
            current_input_word_2 = self.embedding(decoder_current_input_word_2)

            context_1 = self.attention(decoder_current_hidden_state_1[-1], encoder_output)
            context_2 = self.attention(decoder_current_hidden_state_2[-1], encoder_output)

            gru_input_1 = torch.cat([current_input_word_1.squeeze(1), context], dim=1).unsqueeze(1)
            gru_input_2 = torch.cat([current_input_word_2.squeeze(1), context], dim=1).unsqueeze(1)

            gru_output_1, decoder_current_hidden_state_1 = self.gru(gru_input_1, decoder_current_hidden_state_1)
            gru_output_2, decoder_current_hidden_state_2 = self.gru(gru_input_2, decoder_current_hidden_state_2)

            prob_1 = self.to_final_output(gru_output_1.squeeze(1))
            prob_2 = self.to_final_output(gru_output_2.squeeze(1))

            logprob_1 = F.log_softmax(prob_1, dim=1)
            logprob_2 = F.log_softmax(prob_2, dim=1)
            
            decoder_current_input_word_1 = logprob_1.max(1)[1]
            decoder_current_input_word_2 = logprob_2.max(1)[1]
            
            seq_predictions_1 = torch.cat((seq_predictions_1, decoder_current_input_word_1), dim=0)
            seq_predictions_2 = torch.cat((seq_predictions_2, decoder_current_input_word_2), dim=0)
            
            #print(seq_predictions_1)
            #print(seq_predictions_2)
        
        return seq_predictions_1, seq_predictions_2


    def _get_teacher_learning_ratio(self, training_steps): # TODO: change scheduled sampling scheme
        epoch = training_steps
        return max(30 - epoch/2, 0) / 30
        # for epochs 1 ~ 30, ratio is 0.9999 ~ 0.5
        # for epochs larger than 60, ratio is 0



class VideoCaptionGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super(VideoCaptionGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, prev_sentences, mode, curr_sentences=None, steps=None):
        """
        Args:
            param avi_feats(Variable): size(batch size x 80 x 4096)
            param target_sentences: ground truth for training, None for inference
        Returns:
            seq_Prob
            seq_predictions
        """
        top_layer_output, last_time_step_all_layers_output = self.encoder(prev_sentences) # prev_sentences (batch_size, sentence_length)

        if mode == 'train': # (self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', steps=None):
            seq_Prob, seq_predictions = self.decoder(
                encoder_last_hidden_state = last_time_step_all_layers_output,
                encoder_output = top_layer_output,
                targets = curr_sentences,
                mode = mode,
                steps=steps
            )

        elif mode == 'inference':
            seq_Prob, seq_predictions = self.decoder.infer(
                encoder_last_hidden_state= last_time_step_all_layers_output,
                encoder_output= top_layer_output,
            )
        
        elif mode == 'beam_search':
            seq_Prob, seq_predictions = self.decoder.beam_search(
                encoder_last_hidden_state= last_time_step_all_layers_output,
                encoder_output= top_layer_output,
            )
        
        else:
            raise KeyError('mode is not valid')

        return seq_Prob, seq_predictions
        # seq_Prob: (batch, seq_len, vocab_size)
        # seq_predictions: (batch, seq_length)

if __name__ == '__main__':
    from dataset import TrainingDataset, collate_fn
    from vocabulary import Vocabulary
    from torch.utils.data import DataLoader

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
        seq_Prob, seq_predictions = model(prev_sentences=padded_prev_sentences, mode='train', curr_sentences=padded_curr_sentences, steps=step)

        print(seq_Prob)
        print()
        print(seq_predictions)
        break