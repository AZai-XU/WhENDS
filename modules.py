import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, get_laplacian, to_scipy_sparse_matrix, is_undirected
from torch.autograd import Variable, grad
from scipy.sparse.linalg import eigs, eigsh
from einops import rearrange, repeat
from torch_geometric.nn import SAGEConv,GATConv,InnerProductDecoder,global_mean_pool

def _laplacian_ev(data):
    # edge_weight = data.edge_attr
    edge_weight = None
    normalization = None
    edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight, normalization, num_nodes= data.num_nodes)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
    _, ev = eigsh(L, k=1, which='LM', return_eigenvectors=True)
    ev = torch.from_numpy(ev)
    
    return ev

class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, device, act = None, bias = True, dropout = 0.):
        super(GConv, self).__init__()
        self.conv = SAGEConv(input_dim, output_dim).to(device)
        #self.conv = GATConv(input_dim, output_dim, edge_dim=64).to(device)
        self.dropout = dropout
        self.act = act
    
    def forward(self, x, edge_index):
        z = self.conv(x, edge_index)
        if self.act != None:
            z = self.act(z)
        z = F.dropout(z, p = self.dropout, training = self.training)
        return z 
    
class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(GNN, self).__init__()
        self.conv1 = GConv(input_dim, input_dim, act=nn.SiLU(),device=device)
        self.conv2 = GConv(input_dim, output_dim,device=device)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        z = self.conv2(z, edge_index)
        return z

class GRU_2(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, device, bias = True):
        super(GRU_2, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(layer_num):
            if i == 0:
                self.weight_xz.append( nn.Linear(input_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hz.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_xr.append( nn.Linear(input_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hr.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_xh.append( nn.Linear(input_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hh.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias))  
            else:
                self.weight_xz.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hz.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias))  
                self.weight_xr.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hr.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_xh.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
                self.weight_hh.append( nn.Linear(hidden_size, hidden_size, device=self.device, bias= bias)) 
    
    def forward(self, x, h):
        h_out = torch.zeros(h.size()).to(self.device)
        for i in range(self.layer_num):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](x) + self.weight_hz[i](h[i]))
                r_g = torch.sigmoid(self.weight_xr[i](x) + self.weight_hr[i](h[i]))
                h_tilde_g = torch.tanh(self.weight_xh[i](x) + self.weight_hh[i](r_g * h[i]))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](out) + self.weight_hz[i](h[i]))
                r_g = torch.sigmoid(self.weight_xr[i](out) + self.weight_hr[i](h[i]))
                h_tilde_g = torch.tanh(self.weight_xh[i](out) + self.weight_hh[i](r_g * h[i]))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            h_out[i] = out
        return h_out
    
# class Statistic_Predictor(nn.Module):
#     def __init__(self, input_dim, vec_dim, hidden_dim, layer_num, device, act = None, bias = True, dropout = 0.):
#         super(Statistic_Predictor, self).__init__()
#         self.device = device
#         self.input_dim = input_dim
#         self.vec_dim = vec_dim
#         self.hidden_dim = hidden_dim
#         self.gru_stats = GRU_2(input_dim + vec_dim, hidden_dim + vec_dim, layer_num, device = self.device)
#         self.linear_out = nn.Linear(hidden_dim + vec_dim, input_dim + vec_dim, device=self.device, bias= bias)

#     def forward(self,h_stats_prev,mean,cov): 
#         stats = torch.cat([mean, cov],dim = -1)
#         h_stats = self.gru_stats(stats, h_stats_prev)
#         out = self.linear_out(h_stats[-1])
#         next_mean_out, next_cov_out = torch.split(out,[self.input_dim, self.vec_dim],dim=-1)
#         next_cov_out = torch.exp(next_cov_out)

#         return next_mean_out, next_cov_out, h_stats

class Statistic_Predictor(nn.Module):
    def __init__(self, input_dim, vec_dim, hidden_dim, layer_num, device, act = None, bias = True, dropout = 0.):
        super(Statistic_Predictor, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.vec_dim = vec_dim
        self.hidden_dim = hidden_dim
        # self.gru = GRU_2(input_dim + vec_dim, hidden_dim + vec_dim, layer_num, device = self.device)
        self.gru_mean = GRU_2(input_dim, hidden_dim, layer_num, device = self.device)
        self.gru_cov = GRU_2(vec_dim, vec_dim, layer_num, device = self.device)
        # self.mean_proj = nn.Linear(hidden_dim, input_dim, device=self.device, bias= bias)
        # self.cov_proj = nn.Linear(vec_dim, vec_dim, device=self.device, bias= bias)

    def forward(self,h_stats_prev,mean,cov): 
        h_mean_prev, h_cov_prev = torch.split(h_stats_prev,[self.input_dim, self.vec_dim],dim=-1)
        h_mean = self.gru_mean(mean,h_mean_prev)
        h_cov = self.gru_cov(cov,h_cov_prev)
        h_stats = torch.cat([h_mean,h_cov],dim=-1)
        # input_vec = torch.cat([mean,cov],dim=-1)
        # h_stats = self.gru(input_vec, h_stats_prev)

        return h_stats
    
class Diff_Predictor(nn.Module):
    def __init__(self, input_dim, vec_dim, hidden_dim, device,expand = 2, act = None, bias = True, dropout = 0.):
        super(Diff_Predictor, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.vec_dim = vec_dim
        self.hidden_dim = hidden_dim
        # self.mean_proj = nn.Linear(hidden_dim + hidden_dim, input_dim, device=self.device, bias= bias)
        # self.cov_proj = nn.Linear(vec_dim + vec_dim, vec_dim, device=self.device, bias= bias)
        self.mean_proj = MLP(hidden_dim + hidden_dim, input_dim,expand*(hidden_dim + hidden_dim), device=self.device, bias= bias)
        self.cov_proj = MLP(vec_dim + vec_dim, vec_dim,expand * (vec_dim + vec_dim), device=self.device, bias= bias)
        # self.skip_mean = torch.ones(1).to(self.device)
        # self.skip_cov = torch.ones(1).to(self.device)
    def forward(self, h_stats_prev, mean, cov):
        h_mean_prev, h_cov_prev = torch.split(h_stats_prev[-1],[self.input_dim, self.vec_dim],dim=-1)
        mean_input = torch.cat([h_mean_prev,mean],dim=-1)
        cov_input = torch.cat([h_cov_prev,cov],dim=-1)
        delta_mean = self.mean_proj(mean_input)
        delta_cov = self.cov_proj(cov_input)
        # pred_mean = delta_mean + self.skip_mean * mean
        # pred_cov = delta_cov + self.skip_cov * cov


        return delta_mean, delta_cov

class FCC(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(FCC,self).__init__()
        self.device = device
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True, device=self.device),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, bias = True):
        super(MLP, self).__init__()
        self.device = device
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size, device=self.device, bias= bias),
            #nn.BatchNorm1d(hidden_size, device=self.device),
            nn.PReLU(device=self.device),
            nn.Linear(hidden_size, output_size, device=self.device, bias= bias)
        )

    def forward(self, x):
        return self.project(x)
    
class MLP_2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP_2, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.project(x)
    
class Contrastive(nn.Module):
    def __init__(self, device, z_dim, window):
        super(Contrastive, self).__init__()
        self.device = device
        self.max_dis = window
        self.linear = nn.Linear(z_dim, z_dim)
    
    def forward(self, all_z, all_node_idx):
        t_len = len(all_node_idx)
        nce_loss = 0
        f = lambda x: torch.exp(x)
        # self.neg_sample = last_h
        for i in range(t_len - self.max_dis):
            for j in range(i+1, i+self.max_dis+1):
                nodes_1, nodes_2 = all_node_idx[i].tolist(), all_node_idx[j].tolist()
                common_nodes = list(set(nodes_1) & set(nodes_2))
                z_anchor = all_z[i][common_nodes]
                z_anchor = self.linear(z_anchor)
                positive_samples = all_z[j][common_nodes]
                pos_sim = f(self.sim(z_anchor, positive_samples, True))
                neg_sim = f(self.sim(z_anchor, all_z[j], False))
                #index = torch.LongTensor(common_nodes).unsqueeze(1).to(self.device)
                neg_sim = neg_sim.sum(dim=-1).unsqueeze(1) #- torch.gather(neg_sim, 1, index)
                nce_loss += -torch.log(pos_sim / (neg_sim)).mean()
                # nce_loss += -(torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1) - torch.gather(neg_sim, 1, index)))).mean()
        return nce_loss / (self.max_dis * (t_len - self.max_dis))   

    def sim(self, h1, h2, pos=False):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        if pos == True:
            return torch.einsum('ik, ik -> i', z1, z2).unsqueeze(1)
        else:
            return torch.mm(z1, z2.t())       
        
