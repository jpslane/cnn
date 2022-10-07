# -*- coding: utf-8 -*-

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    print('Using device:', DEVICE)

!pip install torchdata

import torchtext
import random

def preprocess(review):

    res = []
    for x in review.split(' '):
        remove_beg=True if x[0] in {'(', '"', "'"} else False
        remove_end=True if x[-1] in {'.', ',', ';', ':', '?', '!', '"', "'", ')'} else False
        if remove_beg and remove_end: res += [x[0], x[1:-1], x[-1]]
        elif remove_beg: res += [x[0], x[1:]]
        elif remove_end: res += [x[:-1], x[-1]]
        else: res += [x]
    return res

if __name__=='__main__':
    train_data = torchtext.datasets.IMDB(root='.data', split='train')
    train_data = list(train_data)
    train_data = [(x[0], preprocess(x[1])) for x in train_data]
    train_data, test_data = train_data[0:10000] + train_data[12500:12500+10000], train_data[10000:12500] + train_data[12500+10000:], 

    print('Num. Train Examples:', len(train_data))
    print('Num. Test Examples:', len(test_data))

    print("\nSAMPLE DATA:")
    for x in random.sample(train_data, 5):
        print('Sample text:', x[1])
        print('Sample label:', x[0], '\n')

PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'

from torch.utils import data
from collections import defaultdict

class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):
        self.examples = examples
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    
    def build_dictionary(self): 
        assert self.split == 'train'
        
        # Don't change this
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}

        i = 3
        counter = {}
        print(self.examples)
        for ex in self.examples:
          for word in ex[1]:
            word = word.lower()
            counter[word] = counter.get(word, 0) + 1
            if counter[word] == self.threshold:
              self.idx2word[i] = word
              self.word2idx[word] = i
              i += 1


    def convert_text(self):

        for ex in self.examples:
          converted = []
          for word in ex[1]:
            converted.append(self.word2idx.get(word, 2))
          converted.append(1)
          self.textual_ids.append(converted)

    def get_text(self, idx):
        review = self.textual_ids[idx]
        length = len(review)

        if length < self.max_len:
          review += [0] * (self.max_len - length)
        else:  
          review = review[0:self.max_len]
        return torch.LongTensor(review)

    
    def get_label(self, idx):

        label = {'pos': 1, 'neg': 0}
        return torch.squeeze(torch.LongTensor([(label[self.examples[idx][0]])]))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.get_text(idx), self.get_label(idx)


def sanityCheckDataSet():
    #	Read in the sample corpus
    reviews = [('pos', 'Your life is good when you have money, success and health'),
               ('neg', 'Life is bad when you got not a lot')]
    data = [(x[0], preprocess(x[1])) for x in reviews]
    print("Sample dataset:")
    for x in data: print(x)

    thresholds = [1,2,3]
    print('\n--- TEST: idx2word and word2idx dictionaries ---')
    correct = [[',', '<END>', '<PAD>', '<UNK>', 'a', 'and', 'bad', 'good', 'got', 'have', 'health', 'is', 'life', 'lot', 'money', 'not', 'success', 'when', 'you', 'your'], ['<END>', '<PAD>', '<UNK>', 'is', 'life', 'when', 'you'], ['<END>', '<PAD>', '<UNK>']]
    for i in range(len(thresholds)):
        dataset = TextDataset(data, 'train', threshold=thresholds[i], max_len=3)

        has_passed, message = True, ''
        if has_passed and (dataset.vocab_size != len(dataset.word2idx) or dataset.vocab_size != len(dataset.idx2word)):
            has_passed, message = False, 'dataset.vocab_size (' + str(dataset.vocab_size) + ') must be the same length as dataset.word2idx (' + str(len(dataset.word2idx)) + ') and dataset.idx2word ('+str(len(dataset.idx2word)) +').'
        if has_passed and (dataset.vocab_size != len(correct[i])):
            has_passed, message = False, 'Your vocab size is incorrect. Expected: ' + str(len(correct[i])) + '\tGot: ' + str(dataset.vocab_size)
        if has_passed and sorted(list(dataset.idx2word.keys())) != list(range(0, dataset.vocab_size)):
            has_passed, message = False, 'dataset.idx2word must have keys ranging from 0 to dataset.vocab_size-1. Keys in your dataset.idx2word: ' + str(sorted(list(dataset.idx2word.keys())))
        if has_passed and sorted(list(dataset.word2idx.keys())) != correct[i]:
            has_passed, message = False, 'Your dataset.word2idx has incorrect keys. Expected: ' + str(correct[i]) + '\tGot: ' + str(sorted(list(dataset.word2idx.keys())))
        if has_passed: # Check that word2idx and idx2word are consistent
            widx = sorted(list(dataset.word2idx.items())) 
            idxw = sorted(list([(v,k) for k,v in dataset.idx2word.items()]))
            if not (len(widx) == len(idxw) and all([widx[q] == idxw[q] for q in range(len(widx))])):
                has_passed, message = False, 'Your dataset.word2idx and dataset.idx2word are not consistent. dataset.idx2word: ' + str(dataset.idx2word) + '\tdataset.word2idx: ' + str(dataset.word2idx)

        status = 'PASSED' if has_passed else 'FAILED'
        print('\tthreshold:', thresholds[i], '\tmax_len:', 3, '\t'+status, '\t'+message)
    
    print('\n--- TEST: len(dataset) ---')
    has_passed = len(dataset) == 2
    if has_passed: print('\tPASSED')
    else: print('\tlen(dataset) is incorrect. Expected: 2\tGot: ' + str(len(dataset)))

    print('\n--- TEST: __getitem__(self, idx) ---')
    max_lens = [3,8,15]
    idxes = [0,1]
    combos = [{'threshold': t, 'max_len': m, 'idx': idx} for t in thresholds for m in max_lens for idx in idxes]
    correct = [(torch.tensor([3, 4, 5]), torch.tensor(1)), (torch.tensor([ 4,  5, 15]), torch.tensor(0)), (torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10]), torch.tensor(1)), (torch.tensor([ 4,  5, 15,  7,  8, 16, 17, 18]), torch.tensor(0)), (torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  1,  0,  0]), torch.tensor(1)), (torch.tensor([ 4,  5, 15,  7,  8, 16, 17, 18, 19,  1,  0,  0,  0,  0,  0]), torch.tensor(0)), (torch.tensor([2, 3, 4]), torch.tensor(1)), (torch.tensor([3, 4, 2]), torch.tensor(0)), (torch.tensor([2, 3, 4, 2, 5, 6, 2, 2]), torch.tensor(1)), (torch.tensor([3, 4, 2, 5, 6, 2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 3, 4, 2, 5, 6, 2, 2, 2, 2, 2, 2, 1, 0, 0]), torch.tensor(1)), (torch.tensor([3, 4, 2, 5, 6, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0]), torch.tensor(0)), (torch.tensor([2, 2, 2]), torch.tensor(1)), (torch.tensor([2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2]), torch.tensor(1)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]), torch.tensor(1)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0]), torch.tensor(0))]
    for i in range(len(combos)):
        combo = combos[i]
        dataset = TextDataset(data, 'train', threshold=combo['threshold'], max_len=combo['max_len'])
        returned = dataset.__getitem__(combo['idx'])

        has_passed, message = True, ''
        if has_passed and len(returned) != 2:
            has_passed, message = False, 'dataset.__getitem__(idx) must return 2 things. Got ' + str(len(returned)) +' things instead.'
        if has_passed and (type(returned[0]) != torch.Tensor or type(returned[1]) != torch.Tensor):
            has_passed, message = False, 'Both returns must be of type torch.Tensor. Got: (' + str(type(returned[0])) + ', ' + str(type(returned[1])) + ')'
        if has_passed and (returned[0].shape != correct[i][0].shape):
            has_passed, message = False, 'Shape of first return is incorrect. Expected: ' + str(correct[i][0].shape) + '.\tGot: ' + str(returned[0].shape)
        if has_passed and (returned[1].shape != correct[i][1].shape):
            has_passed, message = False, 'Shape of second return is incorrect. Expected: ' + str(correct[i][1].shape) + '.\tGot: ' + str(returned[1].shape) + '\n\t\tHint: torch.Size([]) means that the tensor should be dimensionless (just a number). Try squeezing your result.'
        if has_passed and (returned[1] != correct[i][1]):
            has_passed, message = False, 'Label (second return) is incorrect. Expected: ' + str(correct[i][1]) + '.\tGot: ' + str(returned[1])
        if has_passed:
            correct_padding_idxes, your_padding_idxes = torch.where(correct[i][0] == 0)[0], torch.where(returned[0] == dataset.word2idx[PAD])[0]
            if not (correct_padding_idxes.shape == your_padding_idxes.shape and torch.all(correct_padding_idxes == your_padding_idxes)):
                has_passed, message = False, 'Padding is not correct. Expected padding indxes: ' + str(correct_padding_idxes) + '.\tYour padding indexes: ' + str(your_padding_idxes)

        status = 'PASSED' if has_passed else 'FAILED'
        print('\tthreshold:', combo['threshold'], '\tmax_len:', combo['max_len'] , '\tidx:', combo['idx'], '\t'+status, '\t'+message)

if __name__ == '__main__':
    sanityCheckDataSet()


if __name__=='__main__':
    train_dataset = TextDataset(train_data, 'train', threshold=10, max_len=150)
    print('Vocab size:', train_dataset.vocab_size, '\n')

    randidx = random.randint(0, len(train_dataset)-1)
    text, label = train_dataset[randidx]
    print('Example text:')
    print(train_data[randidx][1])
    print(text)
    print('\nExample label:')
    print(train_data[randidx][0])
    print(label)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, filter_heights, stride, dropout, num_classes, pad_idx):
        super(CNN, self).__init__()
        
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)

        self.layers = nn.ModuleList()
        for k in filter_heights:
          self.layers.append(nn.Conv2d(1, out_channels, [k, embed_size], stride=stride))
        self.do = nn.Dropout(dropout)
        self.linear = nn.Linear(out_channels * len(self.layers), num_classes)


    def forward(self, texts):
        """
        texts: LongTensor [batch_size, max_len]
        
        Returns output: Tensor [batch_size, num_classes]
        """
        
        embeds = self.embedding(texts)
        
        input_to_conv = torch.unsqueeze(embeds, 1)
        a = []
        for layer in enumerate(self.layers):
          cnn = F.relu(torch.squeeze(layer[1](input_to_conv), 3))
          max = torch.amax(cnn, dim=2)
          a.append(max)

        fin = torch.cat(a, 1)

        hello = self.do(fin)

        return self.linear(hello)

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def sanityCheckModel(all_test_params, NN, expected_outputs, init_or_forward, data_loader):
    print('--- TEST: ' + ('Number of Model Parameters (tests __init__(...))' if init_or_forward=='init' else 'Output shape of forward(...)') + ' ---')
    
    if init_or_forward == "forward":
        for texts_, labels_ in data_loader:
            texts_batch, labels_batch = texts_, labels_
            break

    for tp_idx, (test_params, expected_output) in enumerate(zip(all_test_params, expected_outputs)):       
        if init_or_forward == "forward":
            batch_size = test_params['batch_size']
            texts = texts_batch[:batch_size]

        tps = {k:v for k, v in test_params.items() if k != 'batch_size'}
        stu_nn = NN(**tps)

        if init_or_forward == "forward":
            with torch.no_grad(): 
                stu_out = stu_nn(texts)
            ref_out_shape = expected_output

            has_passed = torch.is_tensor(stu_out)
            if not has_passed: msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
            else: 
                has_passed = stu_out.shape == ref_out_shape
                msg = 'Your Output Shape: ' + str(stu_out.shape)
            

            status = 'PASSED' if has_passed else 'FAILED'
            message = '\t' + status + "\t Init Input: " + str({k:v for k,v in tps.items()}) + '\tForward Input Shape: ' + str(texts.shape) + '\tExpected Output Shape: ' + str(ref_out_shape) + '\t' + msg
            print(message)
        else:
            stu_num_params = count_parameters(stu_nn)
            ref_num_params = expected_output
            comparison_result = (stu_num_params == ref_num_params)

            status = 'PASSED' if comparison_result else 'FAILED'
            message = '\t' + status + "\tInput: " + str({k:v for k,v in test_params.items()}) + ('\tExpected Num. Params: ' + str(ref_num_params) + '\tYour Num. Params: '+ str(stu_num_params))
            print(message)

        del stu_nn


if __name__ == '__main__':
    inputs = [{'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}]
    expected_outputs = [22434, 22531, 22434, 22531, 23874, 23939, 23874, 23939, 41730, 42115, 41730, 42115, 47490, 47747, 47490, 47747, 44578, 44675, 44578, 44675, 47554, 47619, 47554, 47619, 82306, 82691, 82306, 82691, 94210, 94467, 94210, 94467]

    sanityCheckModel(inputs, CNN, expected_outputs, "init", None)
    print()

    inputs = [{'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}]
    expected_outputs = [torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2])]
    sanity_dataset = TextDataset(train_data, 'train', 5, 150)
    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

    sanityCheckModel(inputs, CNN, expected_outputs, "forward", sanity_loader)

if __name__=='__main__':
    THRESHOLD = 5 # Don't change this
    MAX_LEN = 200 # Don't change this
    BATCH_SIZE = 32

    train_dataset = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_dataset = TextDataset(test_data, 'test', THRESHOLD, MAX_LEN, train_dataset.idx2word, train_dataset.word2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


from tqdm.notebook import tqdm

def train_model(model, num_epochs, data_loader, optimizer, criterion):
    print('Training Model...')
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        epoch_acc = 0
        for texts, labels in data_loader:
            texts = texts.to(DEVICE) # shape: [batch_size, MAX_LEN]
            labels = labels.to(DEVICE) # shape: [batch_size]

            optimizer.zero_grad()

            output = model(texts)
            acc = accuracy(output, labels)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, epoch_loss/len(data_loader), 100*epoch_acc/len(data_loader)))
    print('Model Trained!\n')

def accuracy(output, labels):
    """
    Returns accuracy per batch
    output: Tensor [batch_size, n_classes]
    labels: LongTensor [batch_size]
    """
    preds = output.argmax(dim=1) # find predicted class
    correct = (preds == labels).sum().float() # convert into float for division 
    acc = correct / len(labels)
    return acc

if __name__=='__main__':
    cnn_model = CNN(vocab_size = train_dataset.vocab_size, # Don't change this
                embed_size = 128, 
                out_channels = 64, 
                filter_heights = [2, 3, 4], 
                stride = 1, 
                dropout = 0.5, 
                num_classes = 2, # Don't change this
                pad_idx = train_dataset.word2idx[PAD]) # Don't change this

    # Put your model on the device (cuda or cpu)
    cnn_model = cnn_model.to(DEVICE)
    
    print('The model has {:,d} trainable parameters'.format(count_parameters(cnn_model)))

import torch.optim as optim

if __name__=='__main__':    
    LEARNING_RATE = 5e-4

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Define the optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

if __name__=='__main__':    
    N_EPOCHS = 10 # Feel free to change this
    
    # train model for N_EPOCHS epochs
    train_model(cnn_model, N_EPOCHS, train_loader, optimizer, criterion)

import random

def evaluate(model, data_loader, criterion, use_tqdm=False):
    print('Evaluating performance on the test dataset...')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    print("\nSOME PREDICTIONS FROM THE MODEL:")
    iterator = tqdm(data_loader) if use_tqdm else data_loader
    total = 0
    for texts, labels in iterator:
        bs = texts.shape[0]
        total += bs
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)
        
        output = model(texts)
        acc = accuracy(output, labels) * len(labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels) * len(labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if random.random() < 0.0015 and bs == 1:
            print("Input: "+' '.join([data_loader.dataset.idx2word[idx] for idx in texts[0].tolist() if idx not in {data_loader.dataset.word2idx[PAD], data_loader.dataset.word2idx[END]}]))
            print("Prediction:", pred.item(), '\tCorrect Output:', labels.item(), '\n')

    full_acc = 100*epoch_acc/total
    full_loss = epoch_loss/total
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return predictions, full_acc, full_loss

if __name__=='__main__':
    evaluate(cnn_model, test_loader, criterion, use_tqdm=True) # Compute test data accuracy
