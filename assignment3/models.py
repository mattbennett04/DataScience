# the three recurrent networks
#
# all three read the input window one value per time step
# use a tanh hidde layer with a linear output
# the only difference is what goes into the context layer
#   elman - copy of the hidden layer from the previous step
#   jordan - copy of the output from the previous step plus a decaying self link
#   mrn - hidden and output fed back into several memory banks so each bank forgets
#            at a different rate - ulbricht 1994
#
# the prediction is the output at the last step of the window
# context weights are fixed copy links like in the original papers
# only the context -> hidden weights get trained

import torch
import torch.nn as nn


class Elman(nn.Module):

    def __init__(self, hidden, n_in=1):
        super().__init__()
        self.hidden = hidden
        self.in_to_hid = nn.Linear(n_in, hidden)
        self.ctx_to_hid = nn.Linear(hidden, hidden, bias=False)
        self.hid_to_out = nn.Linear(hidden, 1)

    def forward(self, x):
        # x is batch window n_in
        ctx = torch.zeros(x.shape[0], self.hidden)

        for t in range(x.shape[1]):
            h = torch.tanh(self.in_to_hid(x[:, t]) + self.ctx_to_hid(ctx))
            ctx = h  # copy the hidden layer for the next step
        return self.hid_to_out(h).squeeze(-1)


class Jordan(nn.Module):

    # decay controls how much of the old context from teh output is passed on
    # just output isnt a good way to pass on memeory therefore have to control it
    def __init__(self, hidden, n_in=1, decay=0.5):
        super().__init__()
        self.hidden = hidden
        self.decay = decay
        self.in_to_hid = nn.Linear(n_in, hidden)
        self.ctx_to_hid = nn.Linear(1, hidden, bias=False)
        self.hid_to_out = nn.Linear(hidden, 1)

    # make a prediction after reading all inputs
    def forward(self, x):
        ctx = torch.zeros(x.shape[0], 1)

        for t in range(x.shape[1]):
            h = torch.tanh(self.in_to_hid(x[:, t]) + self.ctx_to_hid(ctx))
            y = self.hid_to_out(h)
            ctx = self.decay * ctx + y  # old context fades new output added
        return y.squeeze(-1)


class MultiRecurrent(nn.Module):

    # each ratio is one memory bank new = r * old + 1 - r * feedback
    # r = 0 is a straight copy same as elman/jordan bigger r remembers further back
    def __init__(self, hidden, n_in=1, ratios=(0.0, 0.25, 0.5, 0.75)):
        super().__init__()
        self.hidden = hidden
        self.ratios = ratios
        n_banks = len(ratios)
        self.in_to_hid = nn.Linear(n_in, hidden)
        self.hid_ctx_to_hid = nn.Linear(n_banks * hidden, hidden, bias=False)
        self.out_ctx_to_hid = nn.Linear(n_banks, hidden, bias=False)
        self.hid_to_out = nn.Linear(hidden, 1)

    def forward(self, x):
        batch = x.shape[0]
        n_banks = len(self.ratios)
        r = torch.tensor(self.ratios, dtype=x.dtype)  # one ratio per bank

        hid_ctx = torch.zeros(batch, n_banks, self.hidden)
        out_ctx = torch.zeros(batch, n_banks)
        for t in range(x.shape[1]):
            h = torch.tanh(self.in_to_hid(x[:, t])
                           + self.hid_ctx_to_hid(hid_ctx.reshape(batch, -1))
                           + self.out_ctx_to_hid(out_ctx))
            y = self.hid_to_out(h)
            # each bank keeps r of its old value and takes 1 - r of the new one
            hid_ctx = r[None, :, None] * hid_ctx + (1 - r)[None, :, None] * h[:, None, :]
            out_ctx = r[None, :] * out_ctx + (1 - r)[None, :] * y
        return y.squeeze(-1)


MODELS = {
    "elman": Elman,
    "jordan": Jordan,
    "mrn": MultiRecurrent,
}


def build_model(name, hidden):
    return MODELS[name](hidden)
