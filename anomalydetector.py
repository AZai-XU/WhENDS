import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP, FCC

class Anomalydetector(nn.Module):
    def __init__(self, args):
        super(Anomalydetector, self).__init__()
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
        self.EPS = 1e-15

    def forward(self, stan_h,y_list = [],train = True):
        bce_loss = 0
        score_list = []
        if train == True:
            for t in range(stan_h.__len__()):
                # print("h shape:",stan_h[t].shape)
                # print("y shape:",y_list[t].shape)
                z = self.mlp(stan_h[t])
                # z = stan_h[t]
                edge_score = self.fcc(z)
                edge_score = edge_score.squeeze().float()
                score_list.append(edge_score)
                bce_loss += self._bce_loss(edge_score, y_list[t])
            bce_loss /= stan_h.__len__()

        if train == False:
            for t in range(stan_h.__len__()):
                z = self.mlp(stan_h[t])
                # z = stan_h[t]
                edge_score = self.fcc(z)
                edge_score = edge_score.squeeze().float()
                # print("edge score in model:",edge_score.shape)
                score_list.append(edge_score)
        return bce_loss, score_list

