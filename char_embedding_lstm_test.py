import torch
import numpy


def _sort(_2dtensor, lengths, descending=True):
    sorted_lengths, order = lengths.sort(descending=descending)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    return _2dtensor_sorted_by_lengths, order


class WordCharLSTM(torch.nn.Module):
    def __init__(self,
                 num_word_embeddings,
                 word_embedding_dim,
                 word_lstm,
                 num_char_embeddings,
                 char_embedding_dim=3,
                 num_char_layers=1,
                 num_char_dirs=2,
                 hidden_size=2,
                 char_dropout=0.,
                 word_padding_idx=0,
                 char_padding_idx=0,
                 word_weight=None,
                 char_weight=None):
        super(WordCharLSTM, self).__init__()
        self._word_embeddings = torch.nn.Embedding(
            num_embeddings=num_word_embeddings,
            embedding_dim=word_embedding_dim,
            padding_idx=word_padding_idx,
            _weight=word_weight
        )

        if word_weight is not None:
            self._word_embeddings.weight.requires_grad = False

        self._char_embeddings = torch.nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=char_embedding_dim,
            padding_idx=char_padding_idx,
            _weight=char_weight)
        if char_weight is not None:
            self._char_embeddings.weight.requires_grad = False

        self._word_lstm = word_lstm

        self._char_lstm = torch.nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=hidden_size // num_char_dirs,
            num_layers=num_char_layers,
            bias=True,
            batch_first=True,
            dropout=char_dropout,
            bidirectional=True)

    def forward(self, word_x, char_x):
        # char_x should not be sorted! _char_forward will take care of it
        char_output = self._char_forward(char_x)
        batch_size = word_x.size(0)
        max_seq_len = word_x.max(1)
        char_output = char_output.reshape(batch_size, max_seq_len, -1)  # last dimension is for char lstm hidden size
        embedded = self._word_embeddings(word_x)
        # TODO: concat to word embeddings and do lstm over concatenated vectors!
        embedded = torch.cat([embedded, char_output], -1)

        sequence_lengths = word_x.gt(0).sum(1)  # TODO: maybe send as input?
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths, True)
        packed_output, _ = self._word_lstm(packed)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

    def _char_forward(self, x):
        word_lengths = x.gt(0).sum(1)  # actual word lengths
        sorted_padded, order = _sort(x, word_lengths)
        embedded = self._char_embeddings(sorted_padded)

        word_lengths_copy = word_lengths.clone()
        word_lengths_copy[word_lengths == 0] = 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True)
        packed_output, _ = self._char_lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True)

        _, reverse_sort_order = torch.sort(order, dim=0)
        output = output[reverse_sort_order]

        indices_of_lasts = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indices_of_lasts).squeeze()
        output[word_lengths == 0] = 0
        # return output.reshape(len(batch), max_sentence_length, -1)
        return output


if __name__ == '__main__':
    sentence1 = [
        [1, 1],
        [1, 4, 2, 3, 1],
        [4, 1, 2, 3, 1, 4, 2]
    ]

    sentence2 = [
        [2, 3, 1, 2, 3, 2],
        [3, 3, 4, 4],
        [1, 2],
        [3, 4, 2, 1]
    ]

    batch = [sentence1, sentence2]
    sentence_lengths = [len(sentence) for sentence in batch]
    print(sentence_lengths)
    max_sentence_length = max(sentence_lengths)
    # TODO: replace the name of the variable to padded_sentences accordingly and explain the strange behaviour
    # this is not related but try to avoid [[0]] kind of structure!
    batch = [sentence + [[0]] * (max_sentence_length - len(sentence)) for sentence in batch]
    print(batch)

    word_lengths = [len(word) for sentence in batch for word in sentence]
    max_word_length = max(word_lengths)
    padded_sentences = [word + [0] * (max_word_length - len(word)) for sentence in batch for word in sentence]

    padded_sentences = torch.tensor(padded_sentences)
    word_lengths = padded_sentences.gt(0).sum(1)  # 'actual word lengths'
    print(padded_sentences)

    embedding_tensor = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
    ]).float()

    char_lstm = WordCharLSTM(
        embedding_tensor.size(0),
        embedding_tensor.size(1),
        num_char_layers=1,
        char_weight=embedding_tensor,
        hidden_size=4)
    print(char_lstm(padded_sentences))

    sorted_padded, order = _sort(padded_sentences, word_lengths)
    print(sorted_padded)
    print(word_lengths)
    print(word_lengths[order])

    embeddings = torch.nn.Embedding(
        6,
        3,
        _weight=embedding_tensor)

    embedded = embeddings(sorted_padded)

    lstm = torch.nn.LSTM(
        input_size=3,
        hidden_size=4 // 2,
        num_layers=2,
        bias=True,
        batch_first=True,
        # dropout=DROPOUT,
        bidirectional=True
    )

    word_lengths_copy = word_lengths.clone()
    word_lengths_copy[word_lengths == 0] = 1
    print(word_lengths_copy)
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True)
    packed_output, hidden = lstm(packed)
    output, reconstructed_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True)
    # print(output, output.shape)

    _, reverse_sort_order = torch.sort(order, dim=0)
    output = output[reverse_sort_order]
    # print(output)

    indices = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
    output = output.gather(1, indices).squeeze()
    # print(output)
    output[word_lengths == 0] = 0
    print(output)
    print(output.reshape(len(batch), max_sentence_length, -1))
