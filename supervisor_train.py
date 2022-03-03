import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import copy
from torch.utils.data import DataLoader, TensorDataset

from model import *
from parameters import *
from dataloader_d import *

import warnings
warnings.filterwarnings("ignore")

class ModelSupervisor_train(nn.Module):
    def __init__(self, channel, channel_n, kernel_size, timeslot, scaler_n, resnet_n, step, weather = whether_weather):
        super(ModelSupervisor_train, self).__init__()
        # initialize feature and poi infomation
        self.weather = weather
        if weather == True:
            self.feature = np.load(FEATURE).astype(np.float32)
            if self.feature.max() > 1:
                self.feature = self.feature / np.max(self.feature)
        else:
            self.feature = None
        self.poi = np.load(POI).astype(np.float32)
        if self.poi.max() > 1:
            self.poi = self.poi / np.max(self.poi)

        self.lr = 5e-3
        self.cmplr = 1e-4
        self.epoch = 100
        self.cmpnet = CMPNet(channel, channel, kernel_size, timeslot, self.feature, self.poi)
        self.sr = SRNet(channel, channel_n, scaler_n, resnet_n, step)

    def train(self):
        # restore cmpnet
        net1 = self.cmpnet.to(device)
        net1.load_state_dict(torch.load(cmpnet))
        net2 = self.sr.to(device)

        best_rmse = 1000
        optimizer = optim.Adam([
            {'params': net2.parameters(), 'lr': self.lr, 'betas': (0.9, 0.999)},
            {'params': net1.parameters(), 'lr': self.cmplr, 'betas': (0.9, 0.999)},
        ])
        lr = self.lr
        for epoch in range(self.epoch):
            start = time.time()
            loss_RMSE = []
            loss_MAE = []
            # training
            for i, data in enumerate(zip(pretrain_train, train_train)):
                # start1 = time.time()
                # data[0] - pretrain data and label
                # data[1] - train data and label and ext
                # cmpsr raw mask
                mask_train = copy.deepcopy(data[1][0].to(device))
                mask_train[mask_train != 0] = 1
                label_pre = data[0][1].to(device)
                data_tra = data[1][0].to(device)
                label_tra = data[1][1].to(device)
                ext_tra = data[1][2].to(device)
                try:
                    output1 = net1(data_tra, mask_train)
                    output2 = net2(output1, ext_tra)

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception

                loss_mae_pre = self.mae_loss(output1, label_pre)
                loss_mae = self.mae_loss(output2, label_tra)
                loss_rmse = self.rmse_loss(output2, label_tra)
                loss_MAE.append(loss_mae.item())
                loss_RMSE.append(loss_rmse.item())

                # multi
                total_mae = 0.0001 * loss_mae_pre + 0.1 * loss_mae
                optimizer.zero_grad()
                total_mae.backward(retain_graph=True)
                optimizer.step()

            t = time.time() - start
            result = 'epoch: {}, time: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f} '.format(epoch, t, np.mean(loss_MAE),
                                                                                            np.sqrt(np.mean(loss_RMSE)))
            print(result)
            with open(train_save_txt, "a") as f:
               f.write(result + '\n')

            # evaluating
            if (epoch+1) % 10 == 0:
                jdg, best_rmse = self.evaluate(net1, net2, best_rmse)
                if jdg == True:
                    print("best model!")
                    best_model_2 = net2
                    best_model_1 = net1
                    torch.save(best_model_2.state_dict(), train_save_pkl_sr)
                    torch.save(best_model_1.state_dict(), train_save_pkl_cmp)
                # testing
                self.test(best_model_1, best_model_2)

            if epoch % 20 == 0 and epoch != 0:
               lr /= 2
               optimizer = optim.Adam(net2.parameters(), lr=lr)

        net1.load_state_dict(torch.load(train_save_pkl_cmp))
        net2.load_state_dict(torch.load(train_save_pkl_sr))
        self.test(net1, net2)

    def evaluate(self, net1, net2, b_r):
        jdg = False
        start = time.time()
        loss_RMSE = []
        loss_MAE = []
        for i, (data, label, ext) in enumerate(train_val):
            mask_train = copy.deepcopy(data.to(device))
            mask_train[mask_train != 0] = 1

            data = data.to(device)
            label = label.to(device)
            ext = ext.to(device)

            # (64, 2, 16, 16)
            output1 = net1(data, mask_train)
            output2 = net2(output1, ext)

            loss_mae = self.mae_loss(output2, label)
            loss_rmse = self.rmse_loss(output2, label)
            loss_MAE.append(loss_mae.item())
            loss_RMSE.append(loss_rmse.item())

        t = time.time() - start
        result = 'evaluating -- time: {:.4f}, val_mae: {:.4f}, val_rmse: {:.4f} '.format(t, np.mean(loss_MAE),
                                                                                           np.sqrt(np.mean(loss_RMSE)))
        print(result)
        if np.sqrt(np.mean(loss_RMSE)) < b_r:
            jdg = True
            b_r = np.sqrt(np.mean(loss_RMSE))
        return jdg, b_r

    def test(self, net1, best_model):
        start = time.time()
        loss_RMSE = []
        loss_MAE = []
        for i, (data, label, ext) in enumerate(train_test):
            mask_train = copy.deepcopy(data.to(device))
            mask_train[mask_train != 0] = 1
            data = data.to(device)
            label = label.to(device)
            ext = ext.to(device)

            output1 = net1(data, mask_train)
            output2 = best_model(output1, ext)

            loss_mae = self.mae_loss(output2, label)
            loss_rmse = self.rmse_loss(output2, label)
            loss_MAE.append(loss_mae.item())
            loss_RMSE.append(loss_rmse.item())

        t = time.time() - start
        result = 'testing -- time: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f} '.format(t, np.mean(loss_MAE),
                                                                                        np.sqrt(np.mean(loss_RMSE)))
        print(result)

        with open(train_save_txt, "a") as f:
            f.write(result + '\n')

    def mae_loss(self, y_pred, y_true):
        loss = torch.abs(y_pred - y_true)
        return loss.mean()

    def rmse_loss(self, y_pred, y_true):
        loss = torch.pow((y_pred - y_true), 2)
        return loss.mean()

if __name__ == "__main__":
    # 2x
    model = ModelSupervisor_train(2, 32, 3, 5, 2, 16, 3)
    # 4x
    # model = ModelSupervisor_train(2, 32, 3, 5, 4, 16, 3)
    model.train()



