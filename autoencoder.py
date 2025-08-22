import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, get_laplacian, to_scipy_sparse_matrix, is_undirected
from torch.autograd import Variable, grad
from einops import rearrange, repeat
from torch_geometric.nn import SAGEConv,GATConv,InnerProductDecoder,global_mean_pool
#from ssm import SSM
from glstm import GLSTM_Encoder
from modules import Contrastive, GNN, FCC,MLP
from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import eigs, eigsh

class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(args.z_dim, args.x_dim), nn.ReLU())
        self.gnn = GNN(args.x_dim,args.z_dim,args.device)
        self.emb_proj = MLP(args.z_dim + args.z_dim,args.h_dim,args.h_dim,args.device)
        # self.enc = Temporal_Encoder(args.x_dim, args.h_dim, args.z_dim, args.layer_num, args.device)
        self.enc = GLSTM_Encoder(args.x_dim, args.z_dim, args.h_dim, args.layer_num, args.device)
        self.dec = InnerProductDecoder()
        # self.contrastive = Contrastive(args.device, args.z_dim, args.window)
        self.mse = nn.MSELoss(reduction='mean')

        self.rff_features = 1024  # RFF 维度可以根据需要自行调整
        # rff_w 的形状 [rff_features, z_dim]，对应高斯核所需的随机向量
        rff_w = torch.randn(self.rff_features, args.z_dim)
        # rff_b 的形状 [rff_features]，偏置用来做相位平移
        rff_b = 2 * math.pi * torch.rand(self.rff_features)
        # 将其注册为 buffer（不参与梯度更新）
        self.register_buffer('rff_w', rff_w)
        self.register_buffer('rff_b', rff_b)

        self.eps = args.eps
        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.EPS = 1e-15
        self.EPSS = 1e-6

    def forward(self, dataloader,h_prev = None, prev_edge = None):
        recon_loss = 0
        nce_loss = 0
        all_z = []
        all_h = []
        all_node_idx = []
        ev_list = []
        for t,data in enumerate(dataloader):
            data = data.to(self.device)
            x = data.x
            y = data.y
            #y = data.y.unsqueeze(1).float()
            edge_index = data.edge_index
            # print(edge_index)
            # print("y shape", y.shape)
            # print("edge shape", edge_index.shape)
            node_index = data.node_index  
            if h_prev == None:
                h_prev = torch.zeros(self.layer_num,x.size(0), self.h_dim + self.h_dim + self.h_dim).to(self.device)
            if prev_edge == None:
                prev_edge = edge_index

            z_t, h_t = self.enc(x, edge_index, h_prev, prev_edge)
            emb_t = z_t[edge_index[0]] + z_t[edge_index[1]]
            # emb_t = self.emb_proj(torch.cat([z_t[edge_index[0]], z_t[edge_index[1]]],dim=-1))
            # ###########################
            # test_norm._test_norm(emb_t)
            # ###########################
            all_h.append(emb_t)
            recon_loss += self._recon_loss(z_t, x, edge_index)
            
            h_prev = h_t   
            prev_edge = edge_index
            # all_z.append(z_t)
            # all_node_idx.append(node_index)

        
        recon_loss /= dataloader.__len__()
        # nce_loss /= dataloader.__len__()
        # nce_loss = self.contrastive(all_z, all_node_idx)
        

        return h_prev, prev_edge, recon_loss, nce_loss, all_h
    
    def _recon_loss(self, z, x, pos_edge_index, neg_edge_index=None):        
        x_hat = self.linear(z)
        feature_loss = self.mse(x, x_hat)
        weight = torch.sigmoid(torch.exp(-torch.norm(z[pos_edge_index[0]] - z[pos_edge_index[1]], dim=1, p=2)))
        pos_loss = (-torch.log(self.dec(z, pos_edge_index) + self.EPS)*weight).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        #weight = torch.sigmoid(torch.exp(torch.norm(z[neg_edge_index[0]] - z[neg_edge_index[1]], dim=1, p=2)))
        neg_loss = -torch.log(1 -
                              self.dec(z, neg_edge_index, sigmoid=True) +
                              self.EPS).mean()
        #neg_loss = (-torch.log(1 - self.dec(z, neg_edge_index) + self.EPS)*weight).mean()
        return pos_loss + neg_loss + feature_loss 



    def _white_trans(self, mean, cov, x):
        cov_matrix = cov
        I = torch.eye(cov_matrix.shape[0]).to(self.device)
        cov_matrix = cov_matrix + self.EPSS * I
        x_center = x - mean
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        eigvals = torch.clamp(eigvals, min=self.EPSS)


        whiten_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        whitened_x = x_center @ whiten_matrix

        return whitened_x

    def _get_stats(self, emb_pool):
        # all_edge = torch.cat(emb_pool,dim=0)
        all_edge = emb_pool
        mean = all_edge.mean(dim=0)
        cov = torch.cov(all_edge.T)

        return mean, cov

    