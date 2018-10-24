import torch
import torch.nn as nn

from data_utils import START_TAG_IDX, STOP_TAG_IDX, PAD_IDX

EMBED_SIZE = 300
CHAR_EMBEDDING_DIM = 100
HIDDEN_SIZE = 300
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 1

torch.manual_seed(1)
CUDA = torch.cuda.is_available()


def _sort(_2dtensor, lengths, descending=True):
    sorted_lengths, order = lengths.sort(descending=descending)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    return _2dtensor_sorted_by_lengths, order


class lstm_crf(nn.Module):
    def __init__(
            self,
            vocab_size,
            num_tags,
            embeddings,
            num_char_embeddings,
            char_lstm):
        super(lstm_crf, self).__init__()
        self.lstm = lstm(
            vocab_size,
            num_tags,
            embeddings,
            num_char_embeddings=num_char_embeddings,
            char_embedding_dim=CHAR_EMBEDDING_DIM,
            char_lstm=char_lstm)

        self.crf = crf(num_tags)
        self = self.cuda() if CUDA else self

    def forward(self, word_x, char_x, y):  # for training
        mask = word_x.data.gt(0).float()  # because 0 is pad_idx, doesn't really belong here, I guess
        h = self.lstm(word_x, mask, char_x)
        Z = self.crf.forward(h, mask)  # partition function
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    #TODO : add char lstm part
    def decode(self, word_x, char_x):  # for prediction
        mask = word_x.data.gt(0).float()  # again 0 is probably because of pad_idx, pass mask as parameter
        h = self.lstm(word_x, mask, char_x)
        return self.crf.decode(h, mask)


# TODO: comment all dimension conversions for better understanding!
class lstm(nn.Module):
    def __init__(
            self,
            vocab_size,
            num_tags,
            embeddings,
            num_char_embeddings,
            char_embedding_dim,
            char_lstm,
            char_padding_idx=0):
        super(lstm, self).__init__()

        # architecture
        self.char_embeddings = nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=char_embedding_dim,
            padding_idx=char_padding_idx)

        self.word_embeddings = nn.Embedding(
            vocab_size,
            EMBED_SIZE,
            padding_idx=PAD_IDX,
            _weight=embeddings)

        self.word_embeddings.weight.requires_grad = False

        self.char_lstm = char_lstm

        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE + 200,
            hidden_size=HIDDEN_SIZE // NUM_DIRS,
            num_layers=NUM_LAYERS,
            bias=True,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL)

        self.out = nn.Linear(HIDDEN_SIZE, num_tags)  # LSTM output to tag

    def init_hidden(self, batch_size):  # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)  # hidden states
        c = zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)  # cell states
        return (h, c)

    def forward(self, word_x, mask, char_x):

        char_output = self._char_forward(char_x)
        batch_size = word_x.size(0)
        max_seq_len = word_x.size(1)
        char_output = char_output.reshape(batch_size, max_seq_len, -1)  # last dimension is for char lstm hidden size

        word_x = self.word_embeddings(word_x)

        word_x = torch.cat([word_x, char_output], -1)

        initial_hidden = self.init_hidden(batch_size)  # batch size is first
        word_x = nn.utils.rnn.pack_padded_sequence(word_x, mask.sum(1).int(), batch_first=True)
        output, hidden = self.lstm(word_x, initial_hidden)

        output, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.out(output)  # batch x seq_len x num_tags
        output *= mask.unsqueeze(-1)  # mask - batch x seq_len -> batch x seq_len x 1
        return output

    def _char_forward(self, x):
        word_lengths = x.gt(0).sum(1)  # actual word lengths
        sorted_padded, order = _sort(x, word_lengths)
        # print(sorted_padded)
        embedded = self.char_embeddings(sorted_padded)

        word_lengths_copy = word_lengths.clone()
        word_lengths_copy[word_lengths == 0] = 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True)
        packed_output, _ = self.char_lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True)

        _, reverse_sort_order = torch.sort(order, dim=0)
        output = output[reverse_sort_order]

        indices_of_lasts = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indices_of_lasts).squeeze()
        output[word_lengths == 0] = 0
        # return output.reshape(len(batch), max_sentence_length, -1)
        return output


class crf(nn.Module):
    def __init__(self, num_tags):
        super(crf, self).__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        # TODO: again, check necessity for using start and stop tags explicitly
        self.transition = nn.Parameter(randn(num_tags, num_tags))
        self.transition.data[START_TAG_IDX, :] = -10000.  # no transition to SOS
        self.transition.data[:, STOP_TAG_IDX] = -10000.  # no transition from EOS except to PAD
        self.transition.data[:, PAD_IDX] = -10000.  # no transition from PAD except to PAD
        self.transition.data[PAD_IDX, :] = -10000.  # no transition to PAD except from EOS
        self.transition.data[PAD_IDX, STOP_TAG_IDX] = 0.
        self.transition.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, mask):  # forward algorithm
        # initialize forward variables in log space
        alpha = Tensor(h.shape[0], self.num_tags).fill_(-10000.)  # [B, S]
        # TODO: pytorch tutorial says wrap it in a variable to get automatic backprop, do we need it here? to be checked
        alpha[:, START_TAG_IDX] = 0.

        transition = self.transition.unsqueeze(0)  # [1, S, S]
        for t in range(h.size(1)):  # iterate through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emission = h[:, t].unsqueeze(2)  # [B, S, 1]
            alpha_t = log_sum_exp(alpha.unsqueeze(1) + emission + transition)  # [B, 1, S] -> [B, S, S] -> [B, S]
            alpha = alpha_t * mask_t + alpha * (1 - mask_t)

        Z = log_sum_exp(alpha + self.transition[STOP_TAG_IDX])
        return Z  # partition function

    def score(self, h, y, mask):  # calculate the score of a given sequence
        batch_size = h.shape[0]
        score = Tensor(batch_size).fill_(0.)
        # TODO: maybe instead of unsqueezing following two separately do it after sum in line for score calculation
        # TODO: check if unsqueezing needed at all
        h = h.unsqueeze(3)
        transition = self.transition.unsqueeze(2)
        y = torch.cat([LongTensor([START_TAG_IDX]).view(1, -1).expand(batch_size, 1), y], 1)  # add start tag to begin
        # TODO: the loop can be vectorized, probably
        for t in range(h.size(1)):  # iterate through the sequence
            mask_t = mask[:, t]
            emission = torch.cat([h[i, t, y[i, t + 1]] for i in range(batch_size)])
            transition_t = torch.cat([transition[seq[t + 1], seq[t]] for seq in y])
            score += (emission + transition_t) * mask_t
        # get transitions from last tags to stop tag: use gather to get last time step
        lengths = mask.sum(1).long()
        indices = lengths.unsqueeze(1)  # we can safely use lengths as indices, because we prepended start tag to y
        last_tags = y.gather(1, indices).squeeze()
        score += self.transition[STOP_TAG_IDX, last_tags]
        return score

    def decode(self, h, mask):  # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        backpointers = LongTensor()
        batch_size = h.shape[0]
        delta = Tensor(batch_size, self.num_tags).fill_(-10000.)
        delta[:, START_TAG_IDX] = 0.

        # TODO: is adding stop tag within loop needed at all???
        # pro argument: yes, backpointers needed at every step - to be checked
        for t in range(h.size(1)):  # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            mask_t = mask[:, t].unsqueeze(1)
            # TODO: maybe unsqueeze transition explicitly for 0 dim for clarity
            next_tag_var = delta.unsqueeze(1) + self.transition  # B x 1 x S + S x S
            delta_t, backpointers_t = next_tag_var.max(2)
            backpointers = torch.cat((backpointers, backpointers_t.unsqueeze(1)), 1)
            delta_next = delta_t + h[:, t]  # plus emission scores
            delta = mask_t * delta_next + (1 - mask_t) * delta  # TODO: check correctness
            # for those that end here add score for transitioning to stop tag
            if t + 1 < h.size(1):
                # mask_next = mask[:, t + 1].unsqueeze(1)
                # ending = mask_next.eq(0.).float().expand(batch_size, self.num_tags)
                # delta += ending * self.transition[STOP_TAG_IDX].unsqueeze(0)
                # or
                ending_here = (mask[:, t].eq(1.) * mask[:, t + 1].eq(0.)).view(1, -1).float()
                delta += ending_here.transpose(0, 1).mul(self.transition[STOP_TAG_IDX])  # add outer product of two vecs
                # TODO: check equality of these two again

        # TODO: should we add transition values for getting in stop state only for those that end here?
        # TODO: or to all?
        delta += mask[:, -1].view(1, -1).float().transpose(0, 1).mul(self.transition[STOP_TAG_IDX])
        best_score, best_tag = torch.max(delta, 1)

        # back-tracking
        backpointers = backpointers.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for idx in range(batch_size):
            prev_best_tag = best_tag[idx]  # best tag id for single instance
            length = int(scalar(mask[idx].sum()))  # length of instance
            for backpointers_t in reversed(backpointers[idx][:length]):
                prev_best_tag = backpointers_t[prev_best_tag]
                best_path[idx].append(prev_best_tag)
            best_path[idx].pop()  # remove start tag
            best_path[idx].reverse()

        return best_path


def _sort_and_reverse_sorting(_2dtensor, lengths):
    sorted_lengths, order = lengths.sort(descending=True)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    # do something and reverse
    unsorted = _2dtensor_sorted_by_lengths.new(*_2dtensor.size())
    return unsorted.scatter_(0, order.unsqueeze(1).expand(*_2dtensor.size()), _2dtensor_sorted_by_lengths)


def reverse_sorting_single_vector(_sorted, order):
    unsorted = _sorted.new(*_sorted.size())
    unsorted.scatter_(0, order, _sorted)
    return unsorted


# TODO: check
def reverse_sorting_batch():
    lengths = torch.tensor([len(indices) for indices in indices_list], dtype=torch.long, device=device)
    lengths_sorted, sorted_idx = lengths.sort(descending=True)

    indices_padded = pad_lists(indices, padding_idx, dtype=torch.long, device=device)  # custom function
    indices_sorted = indices_padded[sorted_idx]

    embeddings_padded = self.embedding(indices_sorted)
    embeddings_packed = pack_padded_sequence(embeddings_padded, lengths_sorted.tolist(), batch_first=True)

    h, (h_n, _) = self.lstm(embeddings_packed)

    h, _ = pad_packed_sequence(h, batch_first=True, padding_value=padding_idx)

    # Reverses sorting.
    h = torch.zeros_like(h).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x


def scalar(x):
    return x.view(-1).data.tolist()[0]


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
