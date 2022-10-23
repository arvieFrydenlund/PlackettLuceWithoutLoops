import torch
import numpy as np

from PlackettLuce import PlackettLuceLoss


def make_targets(bs, k, v):
    targets = []
    for i in range(bs):
        targets.append(np.random.choice(v, k, replace=False))
    return torch.from_numpy(np.asarray(targets)).long()


def test_forward_back_with_loop():

    bs = 3
    k = 7
    v = 11

    pl = PlackettLuceLoss()
    logits = torch.rand([bs, v])
    logits.requires_grad = True
    loop_logits = logits.clone()
    pl_targets = make_targets(bs, k, v)

    pl_avg_loss, pl_full_loss, _ = pl(logits, pl_targets, reduce='none')
    print('\t\t', pl_full_loss)

    loop_pl_avg_loss, loop_pl_full_loss, _ = pl.loop_forward(loop_logits, pl_targets)
    print('\t\t', loop_pl_full_loss)

    print(f'Forward all the same: {torch.allclose(pl_full_loss, loop_pl_full_loss)}')

    print('\t\t', pl_avg_loss, loop_pl_avg_loss)

    torch.autograd.backward(pl_avg_loss, inputs=logits)
    torch.autograd.backward(loop_pl_avg_loss, inputs=loop_logits)

    print('\t\t', logits.grad)
    print('\t\t', loop_logits.grad)

    print(f'Backward all the same: {torch.allclose(logits.grad, loop_logits.grad)}')


def test_dynamic_lengths():
    pass

def test_multidim():
    pass


def test_full_ranking():
    pass


def test_target_values():
    pass


def test_orders():
    pass



if __name__ == '__main__':
    test_forward_back_with_loop()
