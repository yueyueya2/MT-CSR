import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

STEP = 5
HORIZON = 0
val_per = 0.2
test_per = 0.1
batch = 16
side = 8

uncom_ratio = ""
whether_weather = True
name = ""
path = ""
DATA_16_uncmp = "" + uncom_ratio + ".npy"
DATA_16_cmp = ""
DATA_32 = ""
A, B = 0, 0
FEATURE = ""
POI = ""

# pretrain saving path
pretrain_save_txt = path + name + "_CMPpretrain_loss_16_" + uncom_ratio + ".txt"
pretrain_save_pkl = path + name + "_CMPpretrain_net_16_"+ uncom_ratio + ".pkl"

# train saving path
train_save_txt = path + name + "_SRtrain_loss_16_" + uncom_ratio +  ".txt"
train_save_pkl_sr = path + name +  "_SRtrain_net_16_" + uncom_ratio +  ".pkl"
train_save_pkl_cmp = path + name +  "_CMPpretrain_net_16_" + uncom_ratio +  ".pkl"

cmpnet = path + name +  "_CMPpretrain_net_16_" + uncom_ratio +  ".pkl"



