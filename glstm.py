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
from modules import GConv, MLP

class GLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, device, bias = True):
        super(GLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.w_xz = []
        self.w_hz = []
        self.w_xi = []
        self.w_hi = []
        self.w_xf = []
        self.w_hf = []
        self.w_xo = []
        self.w_ho = []
        
        for i in range(layer_num):
            if i == 0:
                self.w_xz.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hz.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.w_xi.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hi.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.w_xf.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hf.append(GConv(hidden_size, hidden_size, device=device, bias=bias))
                self.w_xo.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_ho.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
            else:
                self.w_xz.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hz.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.w_xi.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hi.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.w_xf.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_hf.append(GConv(hidden_size, hidden_size, device=device, bias=bias))
                self.w_xo.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.w_ho.append(GConv(hidden_size, hidden_size, device=device, bias=bias))  
    
    def forward(self, x, edge_index, state, prev_edge):
        state_out = torch.zeros(state.size()).to(self.device)
        for i in range(self.layer_num):
            if i == 0:
                h, c , n= torch.split(state[i],[self.hidden_size, self.hidden_size, self.hidden_size],dim=-1)
                f_t = torch.sigmoid(self.w_xf[i](x, edge_index) + self.w_hf[i](h, prev_edge))
                i_t = torch.sigmoid(self.w_xi[i](x, edge_index) + self.w_hi[i](h, prev_edge))
                o_t = torch.sigmoid(self.w_xo[i](x, edge_index) + self.w_ho[i](h, prev_edge))
                z_t = torch.tanh(self.w_xz[i](x, edge_index) + self.w_hz[i](h, prev_edge))
                c_t = (f_t * c) + (i_t * z_t)
                n_t = (f_t * n) + (i_t)
                # h_t = (o_t) * ((c_t) / (n_t))
                h_t = (o_t) * ((c_t))
            else:
                h, c , n= torch.split(state[i],[self.hidden_size, self.hidden_size, self.hidden_size],dim=-1)
                f_t = torch.sigmoid(self.w_xf[i](x, edge_index) + self.w_hf[i](h, prev_edge))
                i_t = torch.sigmoid(self.w_xi[i](x, edge_index) + self.w_hi[i](h, prev_edge))
                o_t = torch.sigmoid(self.w_xo[i](x, edge_index) + self.w_ho[i](h, prev_edge))
                z_t = torch.tanh(self.w_xz[i](x, edge_index) + self.w_hz[i](h, prev_edge))
                c_t = (f_t * c) + (i_t * z_t)
                n_t = (f_t * n) + (i_t)
                # h_t = (o_t) * ((c_t) / (n_t))
                h_t = (o_t) * ((c_t))
            state_out[i] = torch.cat([h_t,c_t,f_t],dim=-1)
        return state_out

class GLSTM_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_num, device, act = None, bias = True, dropout = 0.):
        super(GLSTM_Encoder, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.conv1 = GConv(input_dim, output_dim, device = device, act=F.relu)
        self.conv2 = GConv(output_dim, output_dim, device = device)
        # self.gru = Graph_GRU(input_dim, hidden_dim, layer_num, device= device)
        self.lstm = GLSTM(output_dim, hidden_dim, layer_num, device= device)
        self.mlp = MLP(input_dim, hidden_dim,hidden_dim, device=device, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, output_dim, device = device, bias=bias)

    
    def forward(self, x, edge_index, h_prev, prev_edge):
        z = self.conv1(x, edge_index)
        z = self.conv2(z, edge_index)
        h = self.lstm(z, edge_index, h_prev, prev_edge)
        out, c , n = torch.split(h[-1,:,:],[self.hidden_dim, self.hidden_dim, self.hidden_dim],dim=-1)
        # out = self.out_proj(out)
        return out, h