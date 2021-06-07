import time

import numpy as np
import pandas as pd

# from lib import utils
import torch
import copy

from torch import nn, optim
from model import *

from parameters import *
from dataloader_d import *
from urbanpy_model import *

class ModelSupervisor_pretrain(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, timeslot, weather = whether_weather):
        super(ModelSupervisor_pretrain, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.timeslot = timeslot
        # initialize feature and poi infomation
        if weather == True:
            self.feature = np.load(FEATURE).astype(np.float32)
            if self.feature.max() > 1:
                self.feature = self.feature / np.max(self.feature)
        else:
            self.feature = None
        self.poi = np.load(POI).astype(np.float32)
        if self.poi.max() > 1:
            self.poi = self.poi / np.max(self.poi)

        self.lr = 5e-4

    def train(self):
        Net = CMPNet(self.in_channel, self.out_channel, self.kernel_size, self.timeslot, self.feature, self.poi)
        Net = Net.to(device)
        optimizer=optim.Adam(Net.parameters(), lr = self.lr)
        lr = self.lr
        best_rmse = 1000
        for epoch in range(100):
            start = time.time()
            loss_RMSE = []
            loss_MAE = []
            # training
            for i, (data, label) in enumerate(pretrain_train):
                mask_train = copy.deepcopy(data.to(device))
                mask_train[mask_train != 0] = 1
                data = data.to(device)
                label = label.to(device)

                output = Net(data, mask_train)

                loss_mae = self.mae_loss(output, label)
                loss_rmse = self.rmse_loss(output, label)
                loss_MAE.append(loss_mae.item())
                loss_RMSE.append(loss_rmse.item())

                optimizer.zero_grad()
                loss_rmse.backward()
                optimizer.step()
            t = time.time() - start
            result = 'epoch: {}, time: {:.4f}, train_mae: {:.4f}, train_rmse: {:.4f} '.format(epoch, t, np.mean(loss_MAE), np.sqrt(np.mean(loss_RMSE)))

            with open(pretrain_save_txt, "a") as f:
               f.write(result + '\n')
            print(result)

            # evaluating
            if (epoch+1) % 10 == 0:
                jdg, best_rmse = self.evaluate(Net, best_rmse)
                if jdg == True:
                    print("best model!")
                    best_model = Net

            if epoch % 15 == 0 and epoch != 0:
                lr /= 2
                optimizer = optim.Adam(Net.parameters(), lr=lr)

        # testing
        self.test(best_model)
        torch.save(best_model.state_dict(), pretrain_save_pkl)

    def evaluate(self, Net, b_r):
        jdg = False
        start = time.time()
        loss_RMSE_val = []
        loss_MAE_val = []
        for i, (data, label) in enumerate(pretrain_val):
            mask_train = copy.deepcopy(data.to(device))
            mask_train[mask_train != 0] = 1
            data = data.to(device)
            label = label.to(device)

            output = Net(data, mask_train)

            loss_mae_val = self.mae_loss(output, label)
            loss_rmse_val = self.rmse_loss(output, label)
            loss_MAE_val.append(loss_mae_val.item())
            loss_RMSE_val.append(loss_rmse_val.item())

        t = time.time() - start
        result = 'evaluating -- time: {:.4f}, val_mae: {:.4f}, val_rmse: {:.4f} '.format(t, np.mean(loss_MAE_val),
                                                                                           np.sqrt(
                                                                                               np.mean(loss_RMSE_val)))
        with open(pretrain_save_txt, "a") as f:
           f.write(result + '\n')
        print(result)

        if np.sqrt(np.mean(loss_RMSE_val)) < b_r:
            jdg = True
            b_r = np.sqrt(np.mean(loss_RMSE_val))
        return jdg, b_r

    def test(self, net):
        start = time.time()
        loss_RMSE_test = []
        loss_MAE_test = []
        for i, (data, label) in enumerate(pretrain_test):
            mask_train = copy.deepcopy(data.to(device))
            mask_train[mask_train != 0] = 1
            data = data.to(device)
            label = label.to(device)

            output = net(data, mask_train)

            loss_mae_test = self.mae_loss(output, label)
            loss_rmse_test = self.rmse_loss(output, label)
            loss_MAE_test.append(loss_mae_test.item())
            loss_RMSE_test.append(loss_rmse_test.item())

        t = time.time() - start
        result = 'testing -- time: {:.4f}, test_mae: {:.4f}, test_rmse: {:.4f} '.format(t, np.mean(loss_MAE_test),
                                                                                        np.sqrt(np.mean(loss_RMSE_test)))
        with open(pretrain_save_txt, "a") as f:
            f.write(result + '\n')
        print(result)
    
    def mae_loss(self, y_pred, y_true):
        mask = (y_true != 0).to(device).float()
        mask /= mask.mean()
        loss = torch.abs(y_pred - y_true)
        loss = loss * mask
        loss[loss != loss] = 0
        return loss.mean()

    def rmse_loss(self, y_pred, y_true):
        mask = (y_true != 0).to(device).float()
        mask /= mask.mean()
        loss = torch.pow((y_pred - y_true), 2)
        loss = loss * mask
        loss[loss != loss] = 0
        return loss.mean()

if __name__ == "__main__":
    pretain = ModelSupervisor_pretrain(in_channel=2, out_channel=2, kernel_size=3, timeslot=5)
    pretain.train()
