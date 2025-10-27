import torch
#from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from torch.optim import Adam, SGD
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


cuda=True
DEVICE = torch.device("cuda" if cuda else "cpu")


def apply_correction(p):
    #p_norm = p/p.sum(dim=-1)
    #p_norm_k = p_norm*k
    p_norm_k = p

    d = (p_norm_k*(1-p_norm_k)).sum(dim=-1,keepdim=True).clamp(min=1e-1)

    correction = p_norm_k/(1-p_norm_k).clamp(min=1e-6)
    correction = correction * torch.exp((0.5 - p_norm_k)/d)
    #correction = correction * (d/2).sqrt()
    #correction = correction * (2/d).sqrt()
    odds = correction
    new_p  = odds/(1+odds).clamp(min=1e-6)
    new_p = new_p#/new_p.sum(dim=-1)
    return new_p


def sample_avg(k,p,mask, correct=False):
    """p: Nxd, k: N, mask, Nxd"""
    #size of domain
    n = mask.sum(dim=-1).unsqueeze(dim=-1)

    s = (p*mask).sum(dim=-1).unsqueeze(dim=-1)
    k = k.unsqueeze(dim=-1)
    #assert((k<=n).all())
    if not (k<=n).all():
      print('error')
      print(k,n)

    k_lt_s = (k<=s).float()#.unsqueeze(dim=-1)

    # if k<=s: p*k/s else: (1-p)(n-k)/(n-s)
    q_p = p*mask
    q_1_p = (1-p)*mask
    q = k_lt_s*(q_p*k/s.clamp(min=1)) + (1-k_lt_s)*(q_1_p*(n-k)/(n-s).clamp(min=1))

    if correct:
      q = apply_correction(q)
    #s = q.clamp(min=0,max=1).bernoulli()
    s = q.nan_to_num(nan=0.0, posinf=0., neginf=0.).clamp(min=0,max=1).bernoulli()


    s = k_lt_s*s + (1-k_lt_s)*(1-s)

    return s*mask

def sample_approx_k_hot(k,p, n_iter=7, correct=False):
    """subtracts if k < r. actual subset size can be more than k"""
    mask = torch.ones_like(p)
    k_in = torch.ones(p.shape[0], device='cuda')*k
    k_in = k_in.to(DEVICE)
    k = k_in
    p_in = p

    S = torch.zeros_like(p)
    factor = torch.ones((p.shape[0],1), device='cuda').to(DEVICE)
    for i in range(n_iter):
        s = sample_avg(k, p, mask, correct=correct)

        #vector so far check sub
        s = (S+factor*s)#.clamp(min=0)
        r=s.int().sum(dim=-1)
        k_lt_r = (k_in < r).float()#.unsqueeze(dim=-1)
        k_eq_r = (k_in==r).float()
        #n = k_lt_r*n + (1-k_lt_r)*(n-r)
        # if k < r choose r-k out of r to undo
        k = k_lt_r*(r-k_in) + (1-k_lt_r)*(k_in-r)
        mask = k_lt_r.unsqueeze(dim=-1)*s.abs() + (1-k_lt_r).unsqueeze(dim=-1)*(1-s.abs())
        #factor = k_lt_r*(-1) + (1-k_lt_r)*1
        factor = k_lt_r*(-1) + (1-k_lt_r)*1
        factor = k_eq_r*0 + (1-k_eq_r)*factor
        factor = factor.unsqueeze(dim=-1)
        # if k < r choose samples to remove from 1-p
        p = k_lt_r.unsqueeze(dim=-1)*(1-p_in)+ (1-k_lt_r.unsqueeze(dim=-1))*p_in

        S = s



    return S

