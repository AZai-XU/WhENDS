import os
import time
import numpy as np
import random
from tqdm import tqdm
import torch
import modules
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import *
from dataset import  DynamicGraphAnomaly
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def edge_inject(dataloader,device):
    new_data = []
    for t,data in enumerate(dataloader):
        data.to(device)
        x = data.x
        edge_index = data.edge_index
        y = data.y
        node_index = data.node_index

        edge_num = edge_index.size(1)
        num_neg_samples = int(random.uniform(0, 0.1) * edge_num)
        negative_edge = negative_sampling(edge_index, num_neg_samples= num_neg_samples).to(device)
        negative_label = torch.zeros(negative_edge.size(1)).to(device)

        new_edges = torch.cat([edge_index, negative_edge], dim=-1).long()
        new_labels = torch.cat([y, negative_label],dim=-1)
        new_edge_attr = torch.randn(len(new_edges), 64)
        newdata = Data(x=x, edge_index=new_edges,node_index=node_index,y=new_labels, edge_attr=new_edge_attr)
        new_data.append(newdata)

    return new_data

def nsem_train(autoencoder, nsem_model,nsem_optimizer, dataloader, args,testdata, tb = None):

    new_data = edge_inject(dataloader,args.device)
    _, _, _, _, all_h = autoencoder(new_data)
    all_h_copy = [tensor.detach() for tensor in all_h] 
    y_list = []
    _, _, _, _, test_h = autoencoder(testdata)
    y_test = []
    for t,data in enumerate(testdata):
        y_test.append(data.y)
    for t,data in enumerate(new_data):
        y_list.append(data.y)
    for es_epoch in tqdm(range(args.es_epochs)):
        with torch.autograd.set_detect_anomaly(True): 
            h_stats_prev, stats_loss,kld_loss, _ = nsem_model(all_h_copy,anomalydetector = None,all_y=y_list)
            nsem_loss = (
                stats_loss
                )
            if tb is not None:
                tb.add_scalar("NSEM/Train_Loss", nsem_loss.item(), es_epoch)
            nsem_optimizer.zero_grad()
            nsem_loss.backward()
            # torch.nn.utils.clip_grad_norm_(sees_model.parameters(), 10)
            nsem_optimizer.step()
        
        if (es_epoch + 1) % 5 == 0:
                
                _, stats_loss_test,_, _ = nsem_model(test_h,anomalydetector = None,all_y=y_test,train = False)
                sees_loss_test = 0
                for t in range(stats_loss_test.__len__()):
                     sees_loss_test += stats_loss_test[t]
                     
                if tb is not None:
                    tb.add_scalar("NSEM/Test_Loss", sees_loss_test.item(), es_epoch)
    
    return h_stats_prev
