import torch
import numpy as np

from PlackettLuce import PlackettLuceLoss


def make_targets(bs, k, v):
    targets = []
    for i in range(bs):
        targets.append(np.random.choice(v, k, replace=False))
    return torch.from_numpy(np.asarray(targets)).long()


def test():

    bs = 3
    k = 7
    v = 11

    pl = PlackettLuceLoss()
    logits = torch.rand([bs, v])
    logits.requires_grad = True
    loop_logits = logits.clone()
    pl_targets = make_targets(bs, k, v)

    pl_avg_loss, pl_loss, pl_num, pl_full_loss = pl(logits, pl_targets)
    print(pl_loss.shape, logits.shape, pl_targets.shape, pl_full_loss.shape)
    print(pl_full_loss)

    loop_pl_avg_loss, loop_pl_loss, loop_pl_num, loop_pl_full_loss = pl.loop_forward(loop_logits, pl_targets)
    print(loop_pl_loss.shape, loop_logits.shape, pl_targets.shape, loop_pl_full_loss.shape)
    print(loop_pl_full_loss)

    print(f'Forward all the same: {torch.allclose(pl_full_loss, loop_pl_full_loss)}')

    print(pl_avg_loss, loop_pl_avg_loss)

    torch.autograd.backward(pl_avg_loss, inputs=logits)
    torch.autograd.backward(loop_pl_avg_loss, inputs=loop_logits)

    print(logits.grad)
    print(loop_logits.grad)

    print(f'Backward all the same: {torch.allclose(logits.grad, loop_logits.grad)}')



if __name__ == '__main__':
    test()
