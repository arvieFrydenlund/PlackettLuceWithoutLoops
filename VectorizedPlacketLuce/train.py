import numpy as np
import torch
from torch import nn
from torch.optim import SGD

from VectorizedPlacketLuce.PlackettLuce import PlackettLuceLoss


#  TODO rank auto encoder?

class RankModel(nn.Module):
    """
    Dumb model which tries to arrange the inputs in order by scoring each item.
    Thus, the embeddings should produce scores that are ordered.
    Note that the inputs are the classes, so this should be easy.
    """
    def __init__(self, num_class=19, max_k=11, dim=256):
        super(RankModel, self).__init__()
        self.num_class = num_class
        self.max_k = max_k
        self.dim = dim

        self.embeddings = nn.Embedding(num_class, dim)
        self.score = nn.Linear(dim, 1, bias=False)
        nn.init.uniform_(self.embeddings.weight, -.01, .01)

    def forward(self, inputs):
        """
        :param inputs: [bs, max_k < num_class]
        :return: [bs, max_k] logits which score each input
        """
        return self.score(nn.functional.tanh(self.embeddings(inputs))).squeeze(dim=-1)

    @staticmethod
    def make_inputs_targets(bs, k, v, _print=False):
        inputs = []
        for i in range(bs):
            inputs.append(np.random.choice(v, k, replace=False))
        inputs = torch.from_numpy(np.asarray(inputs)).long().cuda()
        # inputs, _ = torch.sort(inputs)  # make what I am optimizing more clear but then the targets are in order
        _, targets = torch.sort(inputs, dim=-1)

        if _print:
            print(inputs)
            print(targets)
            exit()

        return inputs, targets


def train():
    num_class = 19
    max_k = 11  # number of rank targets
    dim = 64
    bs = 10

    model = RankModel(num_class, max_k, dim).cuda()
    optimizer = SGD(model.parameters(), lr=0.01)

    pl = PlackettLuceLoss()

    for i in range(90):  # larger values might diverge due to numerical stability issue with eta
        inputs, targets = model.make_inputs_targets(bs, max_k+1, num_class)
        logits = model(inputs)

        # print(inputs[:1, :])
        # print(targets[:1, :])
        # print(logits[:1, :])

        loop_pl_avg_loss, loop_pl_full_loss, _ = pl.loop_forward(logits, pl_targets=targets)

        pl_avg_loss, pl_full_loss, _ = pl(logits, pl_targets=targets)

        print(loop_pl_full_loss[:1, :])
        print(pl_full_loss[:1, :])

        ce_loss = nn.functional.cross_entropy(logits, targets[..., 0])
        # ce_loss.backward()

        pl_avg_loss.backward()
        optimizer.step()

        print(f'Loss at {i}: {pl_avg_loss.item()}, {ce_loss.item()}')
        # print(f'Loss at {i}: {ce_loss.item()}')
        print()


if __name__ == '__main__':
    train()
