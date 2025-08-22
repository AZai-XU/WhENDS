import random
import math
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, get_laplacian, to_scipy_sparse_matrix, is_undirected
from torch.autograd import Variable, grad
from einops import rearrange, repeat
from torch_geometric.nn import SAGEConv,GATConv,InnerProductDecoder,global_mean_pool
#from ssm import SSM
from modules import FCC, MLP
from sklearn.metrics import roc_auc_score

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.mlp = MLP(args.h_dim,args.h_dim,args.h_dim,args.device)
        self.fcc = FCC(args.z_dim, 1, args.device)
        self._bce_loss = nn.BCELoss(reduction='mean')

        self.eps = args.eps
        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.r_dim = args.r_dim
        self.window_size = args.len
        self.EPS = 1e-6

    def forward(self, all_h): 
        true_score_list = []
        fake_score_list = []
        for t in range(all_h.__len__()):
            current_mean, current_cov = self._get_stats(all_h[t])
            # stan_emb = (all_h[t] - current_mean) / current_std
            stan_emb = self._white_trans(current_mean, current_cov, all_h[t])
            # stan_emb = all_h[t]
            mean = torch.zeros_like(stan_emb)
            std = torch.ones_like(stan_emb)
            normal_dist = torch.normal(mean, std)
            z1 = self.mlp(normal_dist)
            true_score = self.fcc(z1)
            true_score_list.append(true_score)
            z2 = self.mlp(stan_emb)
            fake_score = self.fcc(z2)
            fake_score_list.append(fake_score)
            
        return true_score_list, fake_score_list


    def _get_stats(self, emb_pool):
        # all_edge = torch.cat(emb_pool,dim=0)
        all_edge = emb_pool
        mean = all_edge.mean(dim=0)
        cov = torch.cov(all_edge.T)

        return mean, cov
    
    def _cholesky(self, cov):
        I = torch.eye(cov.shape[0]).to(self.device)
        cov = cov + self.EPS * I
        L = torch.linalg.cholesky(cov)
        L_idx = torch.tril_indices(row=L.size(0), col=L.size(1), offset=0)
        L_vec = L[L_idx[0], L_idx[1]]

        return L_vec
    
    def _restore_L(self, L_vec):
        L = torch.zeros(self.h_dim, self.h_dim, device=self.device)
        L_idx = torch.tril_indices(row=self.h_dim, col=self.h_dim, offset=0)
        L[L_idx[0], L_idx[1]] = L_vec

        return L
    
    def _white_trans(self, mean, cov, x):
        cov_matrix = cov
        I = torch.eye(cov_matrix.shape[0]).to(self.device)
        cov_matrix = cov_matrix + self.EPS * I
        x_center = x - mean
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        eigvals = torch.clamp(eigvals, min=self.EPS)


        whiten_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        whitened_x = x_center @ whiten_matrix

        return whitened_x

