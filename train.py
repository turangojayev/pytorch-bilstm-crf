# encoding: utf-8
import sys
import time
from os.path import isfile

import numpy
import re
from sklearn.utils import shuffle

from data_utils import CoNLLDataset, get_processing_word, load_vocab, get_processing_tag, PAD, \
    TOKEN2IDX, START_TAG, STOP_TAG, load_vocab_vectors
from utils import *


def _load_words_tags_chars(words_file, tags_file, chars_file):
    word2idx = load_vocab(words_file)
    char2idx = load_vocab(chars_file, 1)

    tag2idx = load_vocab(tags_file, 3)  # since we have 3 more tokens, see below
    # TODO: again, check the necessity for these and remove accordingly
    tag2idx[PAD] = TOKEN2IDX[PAD]
    tag2idx[START_TAG] = TOKEN2IDX[START_TAG]
    tag2idx[STOP_TAG] = TOKEN2IDX[STOP_TAG]
    return word2idx, tag2idx, char2idx


def _load_data(training_filename, word2idx, tag2idx, char2idx):
    # TODO: right now it just relies on the fact that there is UNK in vocab, has to make sure it exists
    processing_word = get_processing_word(word2idx,
                                          char2idx,
                                          lowercase=False,
                                          chars=True)
    processing_tag = get_processing_tag(tag2idx)

    word_indices_and_tag_indices = CoNLLDataset(training_filename, processing_word, processing_tag)
    # TODO: maybe use explicit transformation for clarity, do you really understand now what is happening below?
    word_id_lists, tag_id_lists = list(zip(*word_indices_and_tag_indices))
    char_id_lists, word_id_lists = list(zip(*[list(zip(*couple)) for couple in word_id_lists]))
    seq_lengths_in_words = numpy.array([len(i) for i in word_id_lists])
    word_lengths = numpy.array([numpy.array([len(word_char_seq) for word_char_seq in sequence])
                                for sequence in char_id_lists])
    return word_id_lists, tag_id_lists, char_id_lists, seq_lengths_in_words, word_lengths


def batch_generator(*arrays, batch_size=32):
    word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths = arrays
    word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths = \
        shuffle(word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths)

    num_instances = len(word_id_lists)
    batch_count = int(numpy.ceil(num_instances / batch_size))
    from tqdm import tqdm
    prog = tqdm(total=num_instances)
    for idx in range(batch_count):
        startIdx = idx * batch_size
        endIdx = (idx + 1) * batch_size if (idx + 1) * batch_size < num_instances else num_instances
        batch_lengths = seq_length_in_words[startIdx:endIdx]
        batch_maxlen = batch_lengths.max()
        argsort = numpy.argsort(batch_lengths)[::-1].copy()  # without the copy torch complains about negative strides
        char_batch = numpy.array(char_id_lists[startIdx:endIdx])[argsort]
        char_batch = [sentence + ((0,),) * (batch_maxlen - len(sentence)) for sentence in char_batch]

        word_lengths = [len(word) for sentence in char_batch for word in sentence]
        max_word_length = max(word_lengths)
        chars = [word + (0,) * (max_word_length - len(word)) for sentence in char_batch for word in sentence]

        chars = LongTensor(chars)

        words = LongTensor([word_ids + (PAD_IDX,) * (batch_maxlen - len(word_ids))
                            for word_ids in word_id_lists[startIdx:endIdx]])
        tags = LongTensor([tag_ids + (PAD_IDX,) * (batch_maxlen - len(tag_ids))
                           for tag_ids in tag_id_lists[startIdx:endIdx]])

        prog.update(len(batch_lengths))
        yield words[argsort], chars, tags[argsort]
    prog.close()


def train(word_vocab_size,
          tag_vocab_size,
          char_vocab_size,
          word_embeddings,
          train_data,
          valid_data,
          epochs=20):
    char_lstm = torch.nn.LSTM(
        input_size=CHAR_EMBEDDING_DIM,
        hidden_size=200 // 2,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True)
    print('char vocab size', char_vocab_size)
    model = lstm_crf(
        word_vocab_size,
        tag_vocab_size,
        torch.tensor(word_embeddings, dtype=torch.float),
        char_vocab_size,
        char_lstm)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # TODO: change naming to something better
    filename = re.sub("\.epoch[0-9]+$", "", '0')
    print(model)
    print("training model...")
    epoch = 0
    for ei in range(epoch + 1, epoch + epochs + 1):
        loss_sum = 0
        timer = time.time()
        batch_count = 0
        model.train()
        for word_x, char_x, y in batch_generator(*train_data):
            model.zero_grad()
            loss = torch.mean(model(word_x, char_x, y))  # forward pass and compute loss
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            loss = scalar(loss)
            loss_sum += loss
            batch_count += 1
        timer = time.time() - timer
        loss_sum /= batch_count
        if ei % SAVE_EVERY and ei != epoch + epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

        loss_sum = 0
        batch_count = 0
        model.eval()
        with torch.no_grad():
            for word_x, char_x, y in batch_generator(*valid_data):
                loss_sum += scalar(torch.mean(model(word_x, char_x, y)))
                batch_count += 1

            print('validation loss: {}'.format(loss_sum / batch_count))


if __name__ == "__main__":
    words_file = "data/words.txt"
    tags_file = "data/tags.txt"
    chars_file = "data/chars.txt"
    valid_file = 'data/more_annotated_test.conll'
    train_file = 'data/more_annotated_train.conll'
    embeddings_file = "/usr/data/audi/w2v/word2vec.txt"
    filtered_embeddings_file = "data/filtered_embeddings.txt"

    word2idx, tag2idx, char2idx = _load_words_tags_chars(words_file, tags_file, chars_file)
    train_data = _load_data(train_file, word2idx, tag2idx, char2idx)
    valid_data = _load_data(valid_file, word2idx, tag2idx, char2idx)
    train(
        len(word2idx),
        len(tag2idx),
        len(char2idx) + 1,
        load_vocab_vectors(filtered_embeddings_file),
        train_data,
        valid_data,
        epochs=100)
