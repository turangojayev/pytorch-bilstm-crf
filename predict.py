from collections import defaultdict

from data_utils import load_vocab_vectors, UNK, _get_spans
from train import _load_words_tags_chars, _load_data, batch_generator
from utils import *


# TODO: model.train should be called with False before evaluation?
def _predict():
    word2idx, tag2idx = _load_words_tags_chars("data/words.txt", "data/tags.txt")
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    embeddings = load_vocab_vectors("data/filtered_embeddings.txt")
    model = lstm_crf(len(word2idx), len(tag2idx), torch.tensor(embeddings, dtype=torch.float))
    model.eval()
    print(model)
    load_checkpoint('0.epoch7', model)

    arrays = _load_data('data/more_annotated_test.conll', None, word2idx, tag2idx)
    type2referenced = defaultdict(set)
    type2predicted = defaultdict(set)

    sentenceIdx = 0
    for x, y in batch_generator(*arrays):
        y_pred = model.decode(x)

        for xi, yi, yp in zip(x, y, y_pred):
            text, predicted_tags = _instance_from(yp, xi, idx2word, idx2tag)
            _, original_tags = _instance_from(yi, xi, idx2word, idx2tag)
            length = int(xi.gt(0).sum()) #TODO: replace check with 0 with check for pad index
            referenced = _get_spans(text[:length], original_tags[:length])
            try:
                predicted = _get_spans(text[:length], predicted_tags[:length])

            except ValueError:
                print('skipping', text[:length])
                print(predicted_tags[:length])
                continue

            for reference in referenced:
                type2referenced[reference[2]].add((sentenceIdx, *reference))

            for prediction in predicted:
                type2predicted[prediction[2]].add((sentenceIdx, *prediction))

            sentenceIdx += 1

    print('recalls')
    for type, referenced in type2referenced.items():
        print(type, len(referenced), len(referenced.difference(type2predicted[type])),
              float(len(referenced.difference(type2predicted[type]))) / len(referenced))

    print()
    print('precisions')
    for type, predicted in type2predicted.items():
        print(type, len(predicted), len(predicted.difference(type2referenced[type])),
              float(len(predicted.difference(type2referenced[type]))) / len(predicted))


def _instance_from(predicted, entry, idx2word, idx2tag):
    words = [idx2word.get(idx.item(), UNK) for idx in entry]
    tags = [idx2tag.get(idx.item()) if isinstance(idx, torch.Tensor) else idx2tag[idx] for idx in predicted]
    return words, tags


if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    _predict()



