import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Statistic_Predictor, Diff_Predictor
import os
from datetime import datetime
from torch.distributions import MultivariateNormal, kl_divergence

class NSEM(nn.Module):
    def __init__(self, args):
        super(NSEM, self).__init__()
        # self.enc = Temporal_Encoder(args.x_dim, args.h_dim, args.z_dim, args.layer_num, args.device)
        self.mse = nn.MSELoss(reduction='mean')

        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.vec_dim = int((args.h_dim + 1)*args.h_dim / 2)
        self.window_size = args.len
        self.EPS = 1e-6
        self.noise_intense = args.noise_intense
        self.diff_predictor = Diff_Predictor(args.z_dim, self.vec_dim, args.h_dim, args.device, expand = args.expand)
        self.stats_predictor = Statistic_Predictor(args.z_dim, self.vec_dim, args.h_dim, args.layer_num, args.device)
        self.proj_head_mean = nn.Linear(self.z_dim, self.z_dim, device = self.device)
        self.proj_head_cov = nn.Linear(self.vec_dim, self.vec_dim, device = self.device)


    def forward(self, all_h, anomalydetector,all_y=None, h_stats_prev = None, train = True):
        if train == True:
            stats_loss = 0
            kld_loss = 0
            all_stan_h = []
            all_mean = []
            all_cov = []
            if h_stats_prev == None:
                h_stats_prev = torch.zeros(self.layer_num, self.h_dim + self.vec_dim).to(self.device)
            for t in range(all_h.__len__()):
                norm_emb = self.split_edge_index(all_h[t],all_y[t])
                true_mean, true_cov_matrix = self._get_stats(norm_emb)
                true_cov_vec = self._cholesky(true_cov_matrix)
                calculated_mean, calculated_cov_matrix = self._get_stats(all_h[t])
                calculated_cov_vec = self._cholesky(calculated_cov_matrix)

                calculated_mean_noise = calculated_mean
                calculated_cov_noise = calculated_cov_vec

                delta_mean, delta_cov_vec = self.diff_predictor(h_stats_prev, calculated_mean_noise, calculated_cov_noise)
                pred_mean = calculated_mean + delta_mean
                pred_cov_vec = calculated_cov_vec + delta_cov_vec

                h_mean_prev, h_cov_prev = torch.split(h_stats_prev[-1],[self.z_dim, self.vec_dim],dim=-1)

                pred_hmean = self.proj_head_mean(h_mean_prev)
                pred_hcov = self.proj_head_cov(h_cov_prev)

            
                h_stats = self.stats_predictor(h_stats_prev, true_mean, true_cov_vec)
                true_delta_mean = true_mean - calculated_mean
                true_delta_cov_vec = true_cov_vec - calculated_cov_vec

                stats_loss += (
                               + self._stats_loss(pred_mean, pred_cov_vec, true_mean, true_cov_vec)
                               )
                
                
                h_stats_prev = h_stats
                stan_emb = self._white_trans(true_mean, true_cov_vec, all_h[t])
                all_stan_h.append(stan_emb)
                all_mean.append(delta_mean)
                all_cov.append(delta_cov_vec)

            stats_loss /= all_h.__len__()
            kld_loss = self._l2_loss(all_mean, all_cov)
            # print(stats_loss)
        if train == False:
            stats_loss = []
            kld_loss = []
            all_stan_h = []
            current_stan_h = []
            if h_stats_prev == None:
                h_stats_prev = torch.zeros(self.layer_num, self.h_dim + self.vec_dim).to(self.device)
            for t in range(all_h.__len__()):
                norm_emb = self.split_edge_index(all_h[t],all_y[t])
                true_mean, true_cov_matrix = self._get_stats(norm_emb)
                true_cov_vec = self._cholesky(true_cov_matrix)

                calculated_mean, calculated_cov_matrix = self._get_stats(all_h[t])
                calculated_cov_vec = self._cholesky(calculated_cov_matrix)

                delta_mean, delta_cov_vec = self.diff_predictor(h_stats_prev, calculated_mean, calculated_cov_vec)
                pred_mean = calculated_mean + delta_mean
                pred_cov_vec = calculated_cov_vec + delta_cov_vec

                stan_emb = self._white_trans(pred_mean, pred_cov_vec, all_h[t])
                all_stan_h.append(stan_emb)

                stats_loss.append(self._stats_loss(pred_mean, pred_cov_vec, true_mean, true_cov_vec))
                kld_loss.append(self._stats_loss(calculated_mean, calculated_cov_vec, true_mean, true_cov_vec))
                
                h_stats = self.stats_predictor(h_stats_prev, pred_mean, pred_cov_vec)
                h_stats_prev = h_stats
        return h_stats_prev, stats_loss,kld_loss, all_stan_h
    
    def _stats_loss(self, mean_pred, cov_pred, normal_mean, normal_cov):
        mean_loss = self.mse(mean_pred, normal_mean)
        cov_loss = self.mse(cov_pred,normal_cov)

        return mean_loss + cov_loss
    
    def _stats_loss_exp(self, mean_pred, cov_pred, normal_mean, normal_cov):
        mean_loss = self.mse(mean_pred, normal_mean)
        cov_loss = self.mse(cov_pred,normal_cov)

        return mean_loss.exp() + cov_loss.exp()
        
    
    def _get_stats(self, emb_pool):
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
        L = self._restore_L(cov)
        cov_matrix = L @ L.T
        x_center = x - mean
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        eigvals = eigvals + self.EPS

        whiten_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        whitened_x = x_center @ whiten_matrix

        return whitened_x
    
    def split_edge_index(self, edge_index, y):
        idx_normal = np.nonzero(y >= 0.8).squeeze()
        normal_edge = edge_index[idx_normal]
        # print("nomral_edge",normal_edge.shape)
        
        return normal_edge