import math

import numpy as np
import torch
from torch import nn as nn


class PlackettLuceLoss(nn.Module):
    """
    Partial Plackett-Luce
    """
    def __init__(self):
        super(PlackettLuceLoss, self).__init__()
        self.ninf = -np.inf

    @staticmethod
    def avg(loss, target_lengths=None, num=None, reduce='none'):
        """
        :param loss: [d1...dn-1, max_k]
        :param target_lengths:
        :param num:
        :return: pl_avg_loss, [1],
                 pl_loss, [d1...dn-1], summed across the ranks
                 [1], number of values
        """
        if num is not None:
            avg_loss = loss.sum() / num
        elif target_lengths is not None:
            num = target_lengths.float().sum()
            avg_loss = loss.sum() / num
        else:
            num = torch.ones([1]) * loss.nelement()
            avg_loss = loss.mean()
        if reduce == 'ranks':
            loss = torch.sum(loss, dim=-1)  # [-1]
        return avg_loss, loss, num

    @staticmethod
    def _is_inf_check(t):  # torch 0.4 does not have isinf
        return bool(t.eq(float('inf')).any() or t.eq(float('-inf')).any())

    @staticmethod
    def _nan_check(t):
        return bool(torch.isnan(t).any())

    @staticmethod
    def bad_loss_check(t):
        is_inf = PlackettLuceLoss._is_inf_check(t)
        is_nan = PlackettLuceLoss._nan_check(t)
        if is_inf or is_nan:
            raise ValueError(f'Bad loss: is inf {is_inf}, is nan {is_nan}, check numerical stability value.')

    def loop_forward(self, logits, pl_targets, reduce='none'):
        """
        Wasteful version that needs to loop over the ranks and recompute the normalization term each iteration.
        For comparison with vectorized version.
        Note, that the most preferred rank will be equal to regular cross-entropy softmax.
        :param logits: [d1...dn-1, vocab_size], float32
        :param pl_targets: [d1...dn-1, max_k], ranked order indices, int64
        :return:  [d1...dn-1, max_k]
        """
        masked_logits = logits.clone()  # we can't modify the actual logits otherwise backwards doesn't make sense.
        pl_full_loss = []
        ninf = torch.ones_like(logits) * self.ninf
        for k in range(pl_targets.shape[-1] - 1):
            pl_full_loss.append(torch.nn.functional.cross_entropy(masked_logits, pl_targets[..., k], reduction='none'))
            masked_logits.scatter_(dim=-1, index=pl_targets[..., k].unsqueeze(dim=-1), src=ninf)
        pl_full_loss.append(torch.nn.functional.cross_entropy(masked_logits, pl_targets[..., -1], reduction='none'))
        pl_full_loss = torch.stack(pl_full_loss, dim=-1)
        return self.avg(pl_full_loss, reduce=reduce)

    def forward(self, logits, pl_targets, target_lengths=None, target_values=None, orders=None,
                reduce='none', **kwargs):
        """
        Note the eta term will make it different from the loop version when the cost becomes very low.
        This will then up weight the lowest ranks, so be careful if you see the loss increasing.
        Otherwise, set eta to zero and do nan check when cost becomes low.

        :param logits:  [d1...dn-1, vocab_size], float32, class scores
        :param pl_targets: [d1...dn-1, max_k], ranked order indices, int64
        :param target_lengths: [d1...dn-1], the dynamic size of k, int64 or None
        :param target_values: [d1...dn-1, max_k], weights, float32
                              Currently we assume that these are normalized to 1 over the ranks.  TODO
        :param orders: [d1...dn-1, max_k], int64, in range (0, max_k) or None
                        these are indices to the orders 'head' i.e. the first element of the order
                        so for example, [0, 1, 1, 3, 3, 5, 5, 5, 5, 9]
                        means the gt is 0, then we have an order of two words, with the head at index 1,
                        then an order of 2 things with the head at 3, then an order of four things with the head at 5.
        :param reduce: str, ('none', 'rank')
        :return: pl_avg_loss, [1],
                 pl_loss, [d1...dn-1], summed across the ranks if reduce = 'ranks'
                          [d1...dn-1, max_k], if reduce = 'none'
                 num [1], number of values (possibly weighted by target_values)
        """

        pl_full_loss, mask = self._forward(logits, pl_targets, target_lengths, orders=orders)
        if target_values is not None:  # weight loss by target values
            pl_full_loss *= target_values.view(-1, target_values.shape[-1])

        if target_values is not None:  # assume normalized target values
            return self.avg(pl_full_loss, mask, num=mask.shape[0], reduce=reduce)
        else:
            return self.avg(pl_full_loss, mask,  reduce=reduce)

    @staticmethod
    def _forward(logits, pl_targets, target_lengths=None, orders=None, eta=0.000001, **kwargs):
        """
        Note that we need to calculate the logits and normalization term, Z, only once, then modify Z
        and renormalize at every instance of the 'loop':

        NLL = - ( sum for i=0...k log(e^w_i) - log(sum w_j in V / {w_0...w_i} e^w_j) )
        =   log(sum w_j in V / {w_0...w_i} e^w_j)  - (sum for i=0...k w_i)

        We use the log-sum-exp trick for numerical stability

        :param logits:  [d1...dn-1, vocab_size], float32
        :param pl_targets: [d1...dn-1, max_k], ranked order indices, int64
        :param target_lengths: [d1...dn-1], the dynamic size of k, int64 or None
        :param orders: [d1...dn-1, max_k], int64, in range (0, max_k) or None
                        these are indices to the orders 'head' i.e. the first element of the order
                        so for example, [0, 1, 1, 3, 3, 5, 5, 5, 5, 9]
                        means the gt is 0, then we have an order of two words, with the head at index 1,
                        then an order of 2 things with the head at 3, then an order of four things with the head at 5.
        :param eta: small float for numerical stability
        :return: [d1...dn-1, max_k], [d1...dn-1, max_k] or None
        """

        max_k = pl_targets.shape[-1]  # max_k needs to be >= 2, otherwise just use cross entropy
        logits = logits.view(-1, logits.shape[-1])  # [-1, vocab_size]
        pl_targets = pl_targets.view(-1, max_k)  # [-1, max_k]

        if target_lengths is not None:
            target_lengths = target_lengths.view(-1)  # [-1]

        m, _ = torch.max(logits, dim=-1, keepdim=True)  # [-1, 1]
        exp_logits = torch.exp(logits - m)  # log-sum-exp trick, must exp before gather

        # rearrange logits by rank order and cut to max_k
        logits_gather = torch.gather(logits, dim=-1, index=pl_targets)  # [-1 max_k]

        # test if it is cheaper to do this gather or to exp logits_gather
        # I think it is, since the exp will only be over a small subset of the vocab
        # logits_exp_gather = torch.gather(exp_logits, dim=-1, index=pl_targets)  # [-1 max_k]
        logits_exp_gather = torch.exp(logits_gather - m)

        pad = torch.zeros(size=[logits.shape[0], 1], device=logits.device, dtype=logits.dtype)
        logits_exp_gather = torch.cat([pad, logits_exp_gather[..., :-1]], dim=-1)  # shift it by a column of zeros

        Z = torch.sum(exp_logits, dim=-1, keepdim=True)  # [-1, 1], from full exp logits not gathered exp logits
        Z_mod = torch.cumsum(logits_exp_gather, dim=-1)  # [-1 max_k], accumulated values of previous logits

        if orders is not None:  # undoes cum sum across order, i.e. selects order head for all elements in partial order
            orders = orders.view(-1, orders.shape[-1])
            orders = torch.min(orders, orders.new_ones([1]) * (max_k - 1))  # mask out pad_id
            orders = orders.view(-1, max_k).long()
            Z_mod = torch.gather(Z_mod, dim=-1, index=orders)

        # need small num for numerical stability, else -inf
        # note this will make it different from the loop method
        # these are only needed for the lower ranks, so we can use smaller eta to make the top ranks the same
        # this is modification after the paper, don't know if it will work in practice TODO
        eta_range = torch.arange(0, max_k, device=logits.device).float()
        eta_range *= (eta / max_k)

        Z = torch.log(Z - Z_mod + eta_range) + m  # [-1 max_k], log renormed Z + log-sum-exp trick
        loss = Z - logits_gather  # [-1 max_k]

        if max_k == logits.shape[-1]:  # fix for full rankings, should be a better way of doing this
            loss.scatter_(-1, index=torch.ones_like(pl_targets)*(max_k-1), src=torch.zeros_like(loss))

        mask = None
        if target_lengths is not None:
            mask = (torch.arange(loss.shape[-1], dtype=torch.int, device=logits.device)[None, :]
                    < target_lengths[:, None].int()).float()

            fill_mask = mask <= 0.0
            loss.masked_fill_(fill_mask, 0.0)
        return loss, mask

