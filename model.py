import torch
import torch.nn as nn

from sequence_labelling.data_utils import START_TAG_IDX, STOP_TAG_IDX, PAD_IDX

EMBED_SIZE = 300
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


class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags, embeddings):
        super(lstm_crf, self).__init__()
        self.lstm = lstm(vocab_size, num_tags, embeddings)
        self.crf = crf(num_tags)
        self = self.cuda() if CUDA else self

    def forward(self, x, y):  # for training
        mask = x.data.gt(0).float()  # because 0 is pad_idx, doesn't really belong here, I guess
        h = self.lstm(x, mask)
        Z = self.crf.forward(h, mask)  # partition function
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    def decode(self, x):  # for prediction
        mask = x.data.gt(0).float()  # again 0 is probably because of pad_idx, pass mask as parameter
        h = self.lstm(x, mask)
        return self.crf.decode(h, mask)


# TODO: comment all dimension conversions for better understanding!
class lstm(nn.Module):
    def __init__(self, vocab_size, num_tags, embeddings):
        super(lstm, self).__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=PAD_IDX, _weight=embeddings)
        self.embed.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE // NUM_DIRS,
            num_layers=NUM_LAYERS,
            bias=True,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags)  # LSTM output to tag

    def init_hidden(self, batch_size):  # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)  # hidden states
        c = zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)  # cell states
        return (h, c)

    def forward(self, x, mask):
        initial_hidden = self.init_hidden(x.shape[0])  # batch size is first
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        output, hidden = self.lstm(x, initial_hidden)
        output, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.out(output)  # batch x seq_len x num_tags
        output *= mask.unsqueeze(-1)  # mask - batch x seq_len -> batch x seq_len x 1
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
