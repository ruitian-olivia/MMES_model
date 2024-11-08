import os
import copy
import pytz
import math
import torch
import sklearn
import datetime
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import Polar
from torch_geometric.loader import DataListLoader, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored

from graph_mIHC_surv_models import mIHC_hypergraph_surv, mIHC_GCN_surv, mIHC_GAT_surv, mIHC_GIN_surv
from surv_clinical_training_function import train_surv_bin, patient_test_surv_bin, setup_seed
from surv_loss_function import NLLSurvLoss
    
class mIHC_CoxGraphDataset(Dataset):
    def __init__(self, surv_df):
        super(mIHC_CoxGraphDataset, self).__init__()
        self.surv_df = surv_df

    def len(self):
        return len(self.surv_df)

    def get(self, idx):
        TMA_ID = self.surv_df['TMA_ID'].tolist()[idx]
        split_name = self.surv_df['split_name'].tolist()[idx]
        match_item = self.surv_df[(self.surv_df["TMA_ID"] == TMA_ID) & (self.surv_df["split_name"] == split_name)]

        survival = match_item['surv_time'].tolist()[0]
        censorship = match_item['censorship'].tolist()[0]
        surv_label = match_item['surv_label'].tolist()[0]

        item = match_item['TMA_ID'].tolist()[0] + "+" + match_item['split_name'].tolist()[0]
        
        mIHC_data_origin = torch.load(match_item['sub_mIHC_graph_path'].tolist()[0]) 

        data = Data(x=mIHC_data_origin.x, edge_index=mIHC_data_origin.edge_index)
        data.survival = torch.tensor(survival)
        data.censorship = torch.tensor(censorship)
        data.surv_label = torch.tensor(surv_label)
        data.item = item
        data.pos_HE = mIHC_data_origin.pos
        
        return data
  
def Train(Argument):
    setup_seed(Argument.random_seed)
    
    model_save_dir = os.path.join("model_weights", Argument.model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    result_save_dir = os.path.join("model_result", Argument.model_name)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
        
    surv_record_dir = os.path.join(result_save_dir,'surv_record')
    if not os.path.exists(surv_record_dir):
        os.makedirs(surv_record_dir)    
    
    surv_csv_path = '../../preprocessed_data/clinical_data/surv_bin_mIHC_patches_df.csv'
    surv_df = pd.read_csv(surv_csv_path)
    
    print("surv_df.shape:", surv_df.shape)
    
    TMA_test1 = Argument.TMA_test1
    TMA_test2 = Argument.TMA_test2
    
    TMA_test_list = [TMA_test1, TMA_test2]
    
    surv_disc_df = surv_df[~surv_df['TMA_ID'].isin(TMA_test_list)]
    surv_test1_df = surv_df[surv_df['TMA_ID'] == TMA_test1]
    surv_test2_df = surv_df[surv_df['TMA_ID'] == TMA_test2]
    
    print("surv_disc_df.shape:", surv_disc_df.shape)
    print("surv_test1_df.shape:", surv_test1_df.shape)
    print("surv_test2_df.shape:", surv_test2_df.shape)

    batch_num = int(Argument.batch_size)
    epochs_num = int(Argument.epochs_num)
    
    train_loss = np.zeros(epochs_num)
    train_CI = np.zeros(epochs_num)
    
    mIHC_TrainDataset = mIHC_CoxGraphDataset(surv_df=surv_disc_df)
    
    mIHC_TestDataset1 = mIHC_CoxGraphDataset(surv_df=surv_test1_df)
    mIHC_TestDataset2 = mIHC_CoxGraphDataset(surv_df=surv_test2_df)
          
    train_loader = DataLoader(mIHC_TrainDataset, batch_size=batch_num, shuffle=True, num_workers=1, pin_memory=True)
    test1_loader = DataLoader(mIHC_TestDataset1, batch_size=batch_num, num_workers=1, pin_memory=True, drop_last=False)
    test2_loader = DataLoader(mIHC_TestDataset2, batch_size=batch_num, num_workers=1, pin_memory=True, drop_last=False)

    if Argument.GNN_type == 'Hypergraph':
        model = mIHC_hypergraph_surv(surv_bins=Argument.surv_bins,
                            mIHC_cell_num=Argument.mIHC_cell_num, 
                            mIHC_dim_target=Argument.mIHC_dim_target,
                            layers=Argument.layers, 
                            surv_mlp=Argument.surv_mlp,
                            dropout=Argument.dropout_rate)
    elif Argument.GNN_type == 'GCN':
        model = mIHC_GCN_surv(surv_bins=Argument.surv_bins,
                            mIHC_cell_num=Argument.mIHC_cell_num, 
                            mIHC_dim_target=Argument.mIHC_dim_target,
                            layers=Argument.layers, 
                            surv_mlp=Argument.surv_mlp,
                            dropout=Argument.dropout_rate)
            
    elif Argument.GNN_type == 'GAT':
        model = mIHC_GAT_surv(surv_bins=Argument.surv_bins,
                            mIHC_cell_num=Argument.mIHC_cell_num, 
                            mIHC_dim_target=Argument.mIHC_dim_target,
                            layers=Argument.layers, 
                            surv_mlp=Argument.surv_mlp,
                            dropout=Argument.dropout_rate)
        
    elif Argument.GNN_type == 'GIN':
        model = mIHC_GIN_surv(surv_bins=Argument.surv_bins,
                            mIHC_cell_num=Argument.mIHC_cell_num, 
                            mIHC_dim_target=Argument.mIHC_dim_target,
                            layers=Argument.layers, 
                            surv_mlp=Argument.surv_mlp,
                            dropout=Argument.dropout_rate)

    device = torch.device('cuda')
    model = model.to(device)
    
    surv_loss = NLLSurvLoss()
    surv_loss = surv_loss.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
                    
    for epoch in range(0, epochs_num):
        train_loss[epoch], train_CI[epoch], train_surv_df\
                = train_surv_bin(model, train_loader, optimizer, surv_loss)

        torch.cuda.empty_cache()
        
        print("Epoch: {:03d}, Train loss: {:.5f}, Train C-index: {:.4f}"\
            .format(epoch+1, train_loss[epoch], train_CI[epoch])) 
      
    model_path="{}/trained_model.pth".format(model_save_dir) 
    torch.save(model.state_dict(), model_path)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter in the model: %.2fM" % (total/1e6))
    
    train_CI_patient, train_surv_df, _ = patient_test_surv_bin(model, train_loader)
    print("[Patient-level] Train C-index: {:.5f}".format(train_CI_patient))
    train_surv_df.to_csv(os.path.join(surv_record_dir, 'train_surv_df.csv'),
        sep=',',index=False,header=True,float_format='%.4f')
    
    ## External test 1 & 2
    test1_CI, test1_surv_df, test1_patient_surv_df = patient_test_surv_bin(model, test1_loader)
    print("[Patient-level] Test1 Patient Number: {}, Test1 C-index: {:.5f}"\
        .format(test1_patient_surv_df.shape[0], test1_CI))
    test1_surv_df.to_csv(os.path.join(surv_record_dir, 'test1_surv_df.csv'),
        sep=',',index=False,header=True,float_format='%.4f')
    test1_patient_surv_df.to_csv(os.path.join(surv_record_dir, 'test1_patient_surv_df.csv'),
        sep=',',index=False,header=True,float_format='%.4f')
    
    test2_CI, test2_surv_df, test2_patient_surv_df = patient_test_surv_bin(model, test2_loader)
    print("[Patient-level] Test2 Patient Number: {} , Test2 C-index: {:.5f}"\
        .format(test2_patient_surv_df.shape[0], test2_CI))
    test2_surv_df.to_csv(os.path.join(surv_record_dir, 'test2_surv_df.csv'),
        sep=',',index=False,header=True,float_format='%.4f')
    test2_patient_surv_df.to_csv(os.path.join(surv_record_dir, 'test2_patient_surv_df.csv'),
        sep=',',index=False,header=True,float_format='%.4f')
    
    t_loss = train_loss[np.where(train_loss > 0)]
    t_CI = train_CI[np.where(train_CI > 0)]
    
    fig = plt.figure(dpi=200,figsize=(20,8),facecolor='white')
    ax = fig.add_subplot(131)
    ax.set_title('Traning Loss')
    ax.plot(range(1,len(t_loss)+1), t_loss, label='Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_ylim(np.min(t_loss)-0.2, np.max(t_loss)+0.2) # consistent scale
    ax.legend()
    
    bx = fig.add_subplot(132)
    bx.set_title('Training C-index')
    bx.plot(range(1,len(t_CI)+1), t_CI, label='Training C-index')
    bx.set_xlabel('Epochs')
    bx.set_ylabel('Value')
    bx.set_ylim(0, 1) # consistent scale
    bx.legend()

    fig.savefig(os.path.join(result_save_dir, 'Training_process.png'), format='png')
    plt.close() 
    
    training_record_dir =  os.path.join(result_save_dir, 'training_record')
    if not os.path.exists(training_record_dir):
        os.makedirs(training_record_dir)   

    np.savetxt(os.path.join(training_record_dir, 'train_loss_record.csv'), train_loss, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(training_record_dir, 'train_CI_record.csv'), train_CI, delimiter=',', fmt='%.4f')

    