import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import torch.nn as nn

import argparse
import os
import matplotlib.pyplot as plt

from models import *

def main(args):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)


    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    # 4.4
    overfit_data = data.TabularDataset.splits(path='data/', train='overfit.tsv', format='tsv', skip_header=True,
                                              fields=[('text', TEXT), ('label', LABELS)])[0]
    overfit_iter, _ = data.BucketIterator.splits((overfit_data, val_data), batch_sizes=(args.batch_size, args.batch_size) , sort_key=lambda x: len(x.text),
                                              device=None, sort_within_batch=True, repeat=False)

    # 4.3

    # net = Baseline(args.emb_dim, vocab)
    net = CNN(args.emb_dim, vocab, args.num_filt, [2,4])
    # net = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    loss_list= []
    accur_list = []
    vloss_list = []
    vaccur_list = []

    # 4.3 and 4.5
    net.train()
    for i in range(args.epochs):
        running_loss = 0.0
        running_accur = 0.0
        for j, samples in enumerate(train_iter, 1):

            batch_input, batch_input_length = samples.text
            #print(batch_input_length)
            labels = samples.label.float()

            optimizer.zero_grad()

            outputs = net(batch_input, batch_input_length)
            # print(inputs, outputs, labels)
            loss = criterion(outputs, labels)
            # print("j,loss", j,loss)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_accur = eval_accur(outputs, labels)
            running_accur += batch_accur

        vrunning_accur = 0.0
        vrunning_loss = 0.0
        for index, k in enumerate(val_iter, 1):

            vbatch, vbatch_length = k.text
            vlabels = k.label.float()
            vout = net(vbatch, vbatch_length)

            vloss = criterion(vout, vlabels)
            vbatch_accur = eval_accur(vout, vlabels)

            vrunning_accur += vbatch_accur
            vrunning_loss += vloss.item()

        loss_list.append(running_loss / j)
        accur_list.append(running_accur / j)
        vaccur_list.append(vrunning_accur / index)
        vloss_list.append(vrunning_loss / index)

    plotresult(loss_list, vloss_list, accur_list, vaccur_list)

    print("finish Training")
    torch.save(net, 'model_cnn.pt')
    net = torch.load('model_cnn.pt')
    net.eval()

    testrunning_accur = 0.0
    for index, k in enumerate(test_iter, 1):
        testbatch, testbatch_length = k.text
        testlabels = k.label.float()
        testout = net(testbatch, testbatch_length)

        testbatch_accur = eval_accur(testout, testlabels)

        testrunning_accur += testbatch_accur
    final_test_accuracy = testrunning_accur / index
    print("Final Test Accuracy", final_test_accuracy)


def plotresult(loss_list, vloss_list, accur_list, vaccur_list):
    plt.plot(loss_list, 'r', vloss_list, 'b')
    plt.xlabel('number of epoches')
    plt.ylabel('loss')
    labels = ['train', 'valid']
    plt.legend(labels)
    plt.title('Loss vs. Epoches')
    plt.show()

    plt.plot(accur_list, 'r', vaccur_list, 'b')
    plt.xlabel('number of epoches')
    plt.ylabel('accuracy')
    labels = ['train', 'valid']
    plt.legend(labels)
    plt.title('Accuracy vs. Epoches')
    plt.show()

def eval_accur(outputs, labels):
    accum_accur = 0.0
    for index, i in enumerate(outputs):
        if i.item() > 0.5:
            r = 1.0
        else:
            r = 0.0
        if r == labels[index].item():
            accum_accur+=1.0

    accum_accur = accum_accur / (index+1)

    return accum_accur
    ######

    # 5 Training and Evaluation

    ######

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--rnn_hidden_dim', type=int, default=100)
    parser.add_argument('--num_filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
