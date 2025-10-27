import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import numpy as np
import sampling.subset_sample as S


cuda=True
DEVICE = torch.device("cuda" if cuda else "cpu")

def sample_approx_k_hot(k,logits,mask=None, st_prob=False, st_norm=True, correct=False):
    """st to prob or logits"""
    p = torch.sigmoid(logits)
    if mask is not None:
      p = p*mask
    k_hot = S.sample_approx_k_hot(k, p, n_iter=5, correct=correct)

    norm_prob = normalized_prob(k, p)


    if st_prob:
        #k_hot = k_hot_st_p(p, k_hot)
        k_hot = k_hot_st_p(norm_prob, k_hot)
        #k_hot = k_hot_st_p(u_st_prob, k_hot)
    else:
        # TODO st with normalized v unnormalized logits
        logits = torch.logit(norm_prob)
        #logits = torch.logit(u_st_prob)
        k_hot = k_hot_st_p(logits, k_hot)

    return k_hot, norm_prob


def normalized_prob(k, p):
    """return normalized prob summing to k for st and logprob computation"""
    n = torch.ones_like(p).sum(dim=-1).unsqueeze(dim=-1)
    k = torch.ones(p.shape[0], device='cuda')*k
    k = k.to(DEVICE)
    s = p.sum(dim=-1).unsqueeze(dim=-1)
    k = k.unsqueeze(dim=-1)

    assert((k<=n).all())

    k_lt_s = (k<=s).float()#.unsqueeze(dim=-1)

    # if k<=s: p*k/s else: (1-p)(n-k)/(n-s)
    q_p = p
    q_1_p = (1-p)
    q = k_lt_s*(q_p*k/s.clamp(min=1)) + (1-k_lt_s)*(q_1_p*(n-k)/(n-s).clamp(min=1))
    #q = q.nan_to_num(nan=0.0, posinf=0., neginf=0.).clamp(min=1e-5,max=1-1e-5)
    q = q.clamp(min=1e-9,max=1-1e-9)


    normalized_p = k_lt_s*q + (1-k_lt_s)*(1-q)

    return normalized_p

def k_hot_st_p(act, k_hot):
    """straight-through unnormalized"""
    return k_hot.detach() + act - act.detach()

def k_hot_st_p_normalized(p, k_hot):
    """TODO straight-through p/1-p normalized to sum to k"""
    pass

