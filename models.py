import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        output = F.sigmoid(output)
        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[1], embedding_dim))

        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # print("x size = sentence length, batch_size", x.size())

        embedded = self.embedding(x)

        # embedded size = sentence length, batch size, embedding_dim

        embedded = embedded.unsqueeze(0)
        # embedded size = 1, sentence length, batch size, embedding_dim

        embedded = embedded.permute(2, 0, 1, 3)
        # embedded size = batch size, 1, sentence length, embedded dim

        C1 = F.relu(self.conv1(embedded))
        C2 = F.relu(self.conv2(embedded))
        # batch size, num_kernels, num_phrase, 1

        C1 = C1.squeeze(3)
        C2 = C2.squeeze(3)

        pool1 = nn.MaxPool1d(C1.size()[2])
        pool2 = nn.MaxPool1d(C2.size()[2])

        C1 = pool1(C1).squeeze(2)
        C2 = pool2(C2).squeeze(2)

        C3 = torch.cat((C1, C2), 1)

        C3 = self.fc1(C3).squeeze(1)

        C3 = F.sigmoid(C3)
        return C3

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, ):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 1)
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        cleaned = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        ret = self.gru(cleaned)[1]
        ret = ret.squeeze(0)
        ret = F.sigmoid(self.fc1(ret))
        ret = ret.squeeze(1)
        return ret



def main():
    pass

if __name__ == '__main__':
    main()