# encoding: utf-8
import sys
import time
from os.path import isfile

import numpy
import re
from sklearn.utils import shuffle

from sequence_labelling.data_utils import CoNLLDataset, get_processing_word, load_vocab, get_processing_tag, PAD, \
    TOKEN2IDX, START_TAG, STOP_TAG, load_vocab_vectors
from utils import *


def _load_words_and_tags(words_file, tags_file):
    word2idx = load_vocab(words_file)

    tag2idx = load_vocab(tags_file, 3)  # since we have 3 more tokens, see below
    tag2idx[PAD] = TOKEN2IDX[PAD]
    tag2idx[START_TAG] = TOKEN2IDX[START_TAG]
    tag2idx[STOP_TAG] = TOKEN2IDX[STOP_TAG]
    return word2idx, tag2idx


def _load_data(training_filename, chars_file, word2idx, tag2idx):
    # TODO: right now it just relies on the fact that there is UNK in vocab, has to make sure it exists
    processing_word = get_processing_word(word2idx,
                                          None,
                                          lowercase=False,
                                          chars=False)
    processing_tag = get_processing_tag(tag2idx)

    word_indices_and_tag_indices = list(CoNLLDataset(training_filename, processing_word, processing_tag))
    word_id_lists = [word_ids for word_ids, _ in word_indices_and_tag_indices]
    tag_id_lists = [tag_id for _, tag_id in word_indices_and_tag_indices]
    lengths = numpy.array([len(i) for i in word_id_lists])
    return word_id_lists, tag_id_lists, lengths


def batch_generator(*arrays, batch_size=32):
    word_id_lists, tag_id_lists, lengths = arrays
    word_id_lists, tag_id_lists, lengths = shuffle(word_id_lists, tag_id_lists, lengths)
    num_instances = len(word_id_lists)
    batch_count = int(numpy.ceil(num_instances / batch_size))
    from tqdm import tqdm
    prog = tqdm(total=num_instances)
    for idx in range(batch_count):
        startIdx = idx * batch_size
        endIdx = (idx + 1) * batch_size if (idx + 1) * batch_size < num_instances else num_instances
        batch_lengths = lengths[startIdx:endIdx]
        batch_maxlen = batch_lengths.max()
        argsort = numpy.argsort(batch_lengths)[::-1].copy()
        # TODO: check necessity for start and stop tags, there are repositories not using them
        X = LongTensor([word_ids + [PAD_IDX] * (batch_maxlen - len(word_ids))
                        for word_ids in word_id_lists[startIdx:endIdx]])
        y = LongTensor([tag_ids + [PAD_IDX] * (batch_maxlen - len(tag_ids))
                        for tag_ids in tag_id_lists[startIdx:endIdx]])

        # print(X.shape, y.shape)
        prog.update(len(batch_lengths))
        yield X[argsort], y[argsort]
    prog.close()


def train(word_vocab_size, tag_vocab_size, embeddings, train_data, valid_data, epochs=20):
    model = lstm_crf(word_vocab_size, tag_vocab_size, torch.tensor(embeddings, dtype=torch.float))
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], model) if len(sys.argv) > 1 and isfile(sys.argv[1]) else 0
    # TODO: change naming to something better
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1] if len(sys.argv) > 1 else '0')
    print(model)
    print("training model...")
    for ei in range(epoch + 1, epoch + epochs + 1):
        loss_sum = 0
        timer = time.time()
        batch_count = 0
        model.train()
        for x, y in batch_generator(*train_data):
            model.zero_grad()
            loss = torch.mean(model(x, y))  # forward pass and compute loss
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
            for x, y in batch_generator(*valid_data):
                loss_sum += scalar(torch.mean(model(x, y)))
                batch_count += 1

            print('validation loss: {}'.format(loss_sum / batch_count))


if __name__ == "__main__":
    # train()
    words_file = "data/words.txt"
    tags_file = "data/tags.txt"
    chars_file = "data/chars.txt"
    valid_file = 'data/more_annotated_test.conll'
    train_file = 'data/more_annotated_train.conll'
    embeddings_file = "/usr/data/audi/w2v/word2vec.txt"
    filtered_embeddings_file = "data/filtered_embeddings.txt"
    word2idx, tag2idx = _load_words_and_tags(words_file, tags_file)
    train_data = _load_data(train_file, None, word2idx, tag2idx)
    valid_data = _load_data(valid_file, None, word2idx, tag2idx)
    train(len(word2idx), len(tag2idx), load_vocab_vectors(filtered_embeddings_file), train_data, valid_data, epochs=100)
