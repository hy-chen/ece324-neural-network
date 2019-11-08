"""
    7. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence

        # pipeline following the 1-7 steps.

        # then check user Interface.

        # check accuracy.
"""
import torch
import torchtext
import torchtext.data as data
import spacy
import readline

def main():

    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='train.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(train_data, val_data, test_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    while True:
        x = input("Enter a sentence\n")

        # use torchtext, create a field called TEXT, store vocabs from input
        # into field, map vocab to GloVe vectors

        # this is a list of string rep of the words in the sentence
        tokens = tokenizer(x)
        # stoi map string to int
        token_ints = [vocab.stoi[tok] for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1) # Shape is [sentence_len, 1]
        # the lengths needed != original length, but tokenized length.
        lengths = torch.Tensor([len(token_ints)])

        models = ['model_baseline.pt', 'model_cnn.pt', 'model_rnn.pt']
        headings = ["Model baseline:", "Model cnn:", "Model rnn:"]
        for index, i in enumerate(models):
            net = torch.load(i)
            net.eval()

            prediction = net(token_tensor, lengths)
            if prediction >=0.5:
                print(headings[index], "subjective", "(", "%.3f" % prediction.item(),")")
            else:
                print(headings[index], "objective", "(", "%.3f" % prediction.item(), ")")

# tokenizer distinguish word-phrase, punctutions and words
def tokenizer(sentence):
    space_en = spacy.load('en')
    # what does .text means here? field or sentence
    return [tok.text for tok in space_en(sentence)]

if __name__ == '__main__':
    main()