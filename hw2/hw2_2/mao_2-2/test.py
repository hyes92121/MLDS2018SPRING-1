import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vocabulary import Vocabulary
from dataset import TestingDataset, collate_fn_test
from model import VideoCaptionGenerator, EncoderRNN, DecoderRNN
from customloss import CustomLoss
from trainer import Trainer
import jieba
import sys
import random

training_data_path='clr_conversation.txt'
helper = Vocabulary(training_data_path)

encoder = EncoderRNN(word_vec_filepath='mao_2-2/word_vectors.npy', hidden_size=256, num_layers=2)
decoder = DecoderRNN(word_vec_filepath='mao_2-2/word_vectors.npy', hidden_size=256, num_layers=2)
model = VideoCaptionGenerator(encoder=encoder, decoder=decoder)

#MDL_OUTDIR = 'two_layer'
#if not os.path.exists(MDL_OUTDIR):
#    os.mkdir(MDL_OUTDIR)

#MDL_PRETRAINED_PATH = MDL_OUTDIR + '/' + 'epoch3_data1920000.pt'
MDL_PRETRAINED_PATH = 'epoch3_data1920000.pt'
if MDL_PRETRAINED_PATH:
	print('Loading pretrained model weights from:', MDL_PRETRAINED_PATH)
	model.load_state_dict(torch.load(MDL_PRETRAINED_PATH))
else:
	print('Fail to load model...')
    
    
    
"""
def parse_testing(sentence, helper, pre_split=False):
    #print(sentence)
    if (not pre_split):
        words = jieba.cut(sentence, cut_all=False)
        #print(words)
        line = ''
        for word in words:
            line = line + word + ' '
            #print(line)
    else:
        line = sentence
    line_add_tokens = helper.reannotate(line)
    #print(line_add_tokens)
    curr_sentence = helper.sentence2index(line_add_tokens)
    #print(curr_sentence)
    return curr_sentence

while(True):
    sentence = input("input : ")
    test_parsed = parse_testing(sentence, helper)
    test_parsed = [test_parsed]
    #print(test_parsed)
    testing_data = torch.Tensor(test_parsed).long().cuda()
    #print(testing_data)
    testing_data = Variable(testing_data)
    model = model.cuda()
    model.eval()
    seq_predictions_1, seq_predictions_2 = model(testing_data, mode='beam_search')
    #print(seq_Prob.topk(2, dim=2, largest=True, sorted=True))
    #test_predictions = seq_predictions[0]

    result = ''
    for s in seq_predictions_1:
        if helper.index2sentence(s)[0] == "<EOS>":
            break;
        result += helper.index2sentence(s)[0]
    print("output_origin : " + result)
    
    result = ''
    for s in seq_predictions_2:
        if helper.index2sentence(s)[0] == "<EOS>":
            break;
        result += helper.index2sentence(s)[0]
    print("output_beam_s : " + result)
"""
'''
f_w = open("evaluation/test_output.txt", 'w')

with open("evaluation/test_input.txt", 'r') as f:
    for idx, line in enumerate(f):
        test_parsed = parse_testing(line, helper, True)
        test_parsed = [test_parsed]
        #print(test_parsed)
        testing_data = torch.Tensor(test_parsed).long().cuda()
        #print(testing_data)
        testing_data = Variable(testing_data)
        model = model.cuda()
        model.eval()
        seq_Prob, seq_predictions = model(testing_data, mode='inference')
        #print(seq_predictions)
        test_predictions = seq_predictions[0]

        result = ''
        for s in test_predictions:
            if helper.index2sentence(s)[0] == "<EOS>":
                break;
            if helper.index2sentence(s)[0] == "<UNK>":
                continue;
            result += helper.index2sentence(s)[0]
        
        f_w.write(result + '\n')
        print("output : " + result)
f_w.close()
'''


if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.cpu()
model.eval()

test_data_in = sys.argv[1]
test_data_out = sys.argv[2]

dataset = TestingDataset(test_data_in, helper, load_into_ram=True)
dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8, collate_fn=collate_fn_test)

f_w = open(test_data_out, 'w')
for batch_idx, batch in enumerate(dataloader): # TODO: currently not using test_input.txt
    # prepare data
    padded_prev_sentences= batch
    if torch.cuda.is_available():
        padded_prev_sentences= padded_prev_sentences.cuda()
    padded_prev_sentences = Variable(padded_prev_sentences)

    # start inferencing process
    seq_Prob, seq_predictions = model(padded_prev_sentences, mode='inference')
            
    for s in seq_predictions:
        tmp = helper.index2sentence(s)
        res = []
        for t in tmp:
            if t == '<EOS>':
                break
            elif t == "<UNK>":
                continue
            res.append(t)
        res = ''.join(res)
        #print(res)
        f_w.write(res + '\n')
    print(batch_idx)
f_w.close()
