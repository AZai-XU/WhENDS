#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import random
from tqdm import tqdm
import torch
import modules
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import *
from dataset import  DynamicGraphAnomaly
from autoencoder import Autoencoder
from nsem import NSEM
from anomalydetector import Anomalydetector
from discriminator import Discriminator
from train import nsem_train
'''
def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)

def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).mean()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).mean()

    return model1_loss, model2_loss
'''

def initialize():
    #torch.manual_seed(0)
    autoencoder = Autoencoder(args).to(args.device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    discriminator = Discriminator(args).to(args.device)
    dc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.dc_lr, weight_decay=args.weight_decay)
    nsem_model = NSEM(args).to(args.device)
    nsem_optimizer = torch.optim.Adam(nsem_model.parameters(), lr=args.es_lr, weight_decay=args.weight_decay)
    anomlaydetector = Anomalydetector(args).to(args.device)
    ad_optimizer = torch.optim.Adam(anomlaydetector.parameters(), lr=args.ad_lr, weight_decay=args.weight_decay)
    return autoencoder, ae_optimizer,discriminator, dc_optimizer, nsem_model, nsem_optimizer, anomlaydetector, ad_optimizer

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    start_time = time.time()
    args = args_parser()

    #tensorboard
    tb_path = f'runs/{args.dataset}/{args.anomaly_ratio}/'
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = SummaryWriter(log_dir=tb_path+f"{time.strftime('%m-%d,%H:%M:%S')}")

    #log
    log_path = f'logs/{args.dataset}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f'{args.anomaly_ratio}')
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")

    #Init dataloader
    dataset = []
    if args.dataset not in ['uci', 'digg', 'btc_alpha', 'btc_otc', 'email', 'as_topology']:
        raise NotImplementedError
    dataset = DynamicGraphAnomaly(root='dataset', name=args.dataset, args=args)
    train_size = dataset.train_size
    train2_size = dataset.train2_size
    data_train = dataset[:train_size]
    data_train2 = dataset[train_size:train2_size + train_size] 
    data_test = dataset[train_size + train2_size:]
    print(len(data_train),len(data_train2),len(data_test))

    #Init labels
    test_labels = []  
    for data in data_test:
        y = data.y
        test_labels.append(y.tolist())
    # test_labels = np.array(test_labels)

    #Start Training

    print("Now start training: autoencoder")
    autoencoder, ae_optimizer,discriminator, dc_optimizer, nsem_model, nsem_optimizer, anomlaydetector, ad_optimizer = initialize()

    load_time = time.time()
    print(f'\n Total Loading Rime:{(load_time-start_time):.4f}')
    max_auc = 0.0   
    max_epoch = 0

    autoencoder.train()
    _bce_loss_ = torch.nn.BCELoss(reduction="mean")
    for ae_epoch in tqdm(range(args.ae_epochs)):
        with torch.autograd.set_detect_anomaly(True):
            h_prev, prev_edge, recon_loss, nce_loss, all_h = autoencoder(data_train)
            ae_loss = (
                args.recon_weight * recon_loss 
            )
            tb.add_scalar("Autoencoder/Recon_Loss", recon_loss.item(), ae_epoch)
            ae_optimizer.zero_grad()
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 10)
            ae_optimizer.step()

    print("Now start training: NSEM")

    nsem_model.train()
    h_stats_prev = nsem_train(autoencoder, nsem_model, nsem_optimizer, data_train,args,testdata=data_test,tb=tb)

    print("Now start training: anomalydetector")

    anomlaydetector.train() 
    y_list = []
    y_test = []
    score_epoch = []

    for t,data in enumerate(data_train2):
        y = data.y.float()
        y_list.append(y)

    for t,data in enumerate(data_test):
        y = data.y.float()
        y_test.append(y)

    _, _, _, _, all_h2 = autoencoder(data_train2)
    all_h2_copy = [tensor.detach() for tensor in all_h2]
    _, _, _,  all_stan_h2 = nsem_model(all_h2_copy,anomlaydetector,all_y = y_list)
    all_stan_h2_copy = [tensor.detach() for tensor in all_stan_h2]
    for epoch in tqdm(range(args.ad_epochs)):
        with torch.autograd.set_detect_anomaly(True):
            bce_loss,_ = anomlaydetector(all_stan_h2_copy,y_list)

            ad_loss = (
                bce_loss 
                )
            
            tb.add_scalar("Anomalydetector/Bce_Loss", bce_loss.item(), epoch)
            ad_optimizer.zero_grad()
            ad_loss.backward()
            torch.nn.utils.clip_grad_norm_(anomlaydetector.parameters(), 10)
            ad_optimizer.step()

        if (epoch+1) % 5 == 0 or epoch == args.ad_epochs-1:
            autoencoder.eval()
            nsem_model.eval()
            anomlaydetector.eval()
            with torch.no_grad():
                _, _, _, _, all_h3 = autoencoder(dataloader = data_test,h_prev = h_prev, prev_edge = prev_edge)
                _, _, _,  all_stan_h3 = nsem_model(all_h3,anomlaydetector, train = False,h_stats_prev = h_stats_prev,all_y=y_test)
                _, score_list = anomlaydetector(all_stan_h3, train = False)
            score_all = []
            label_all = []
            auc_time = []
            score_epoch.append(score_list)

            for t in range(len(score_list)):
                score = score_list[t].cpu().numpy().squeeze()
                score_all.append(score)
                label_all.append(test_labels[t])
                # print("y list in t:",len(test_labels[t]))
                auc = roc_auc_score(test_labels[t], score)
                

            score_all = np.hstack(score_all)
            label_all = np.hstack(label_all)
            auc_all = roc_auc_score(label_all, score_all)
            tb.add_scalar("Anomalydetector/AUC", auc_all.item(), epoch)
            if max_auc <= auc_all:
                max_auc = auc_all
                max_epoch = epoch
    log.writelines(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch},\t")
    log.writelines(f"Last AUC: {auc_all:.4f}\n")
    log.writelines(f"ae_epoch: {args.ae_epochs}\n")
    log.writelines(f"ae_lr: {args.ae_lr}\n")
    log.writelines(f"es_epoch: {args.es_epochs}\n")
    log.writelines(f"es_lr: {args.es_lr}\n")
    log.writelines(f"ad_epoch: {args.ad_epochs}\n")
    log.writelines(f"ad_lr: {args.ad_lr}\n")
    log.writelines(f"sigma: {args.sigma}\n")
    print(f'\n Total Training Rime:{(time.time()-load_time):.4f}')
    print(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}")
    