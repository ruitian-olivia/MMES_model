import os
import cv2
import json
import glob
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import lifelines.utils.concordance as LUC
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from palettable.colorbrewer.diverging import RdYlBu_10_r

from sksurv.metrics import concordance_index_censored

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_surv_bin(model,train_loader,optimizer,surv_loss):
    model.train()
    grad_flag = True
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        EpochSurvloss = 0
        batchcounter = 0
        
        for index, mIHC_data in enumerate(train_loader,1):
            optimizer.zero_grad()          
            
            tempsurvival = mIHC_data.survival  # 单个batch数据中的survival data
            tempcensor = mIHC_data.censorship
            templabel = mIHC_data.surv_label
            
            mIHC_tempID = np.asarray(mIHC_data.item)
                                    
            out_h = model(mIHC_data)
            
            hazards = torch.sigmoid(out_h)
            S = torch.cumprod(1 - hazards, dim=1)
            
            out_risk = -torch.sum(S, dim=1)
            
            loss = surv_loss(h=out_h, y=templabel.to(out_h.device), c=tempcensor.to(out_h.device))

            loss.backward()
            optimizer.step()
            
            Epochloss += loss.cpu().detach().item()
            batchcounter += 1
            
            for riskval, survivalval, censorval in zip(out_risk, tempsurvival, tempcensor):
                EpochRisk.append(riskval.cpu().detach().item())
                EpochSurv.append(survivalval.cpu().detach().item())
                EpochCensor.append(censorval.cpu().detach().item())
            
            for tempID in mIHC_tempID:
                 EpochID.append(tempID)
            
        Epochloss = Epochloss / batchcounter
        Epoch_CI = concordance_index_censored((1-np.array(EpochCensor)).astype(bool), np.array(EpochSurv), np.array(EpochRisk), tied_tol=1e-08)[0]
        
        surv_df = pd.DataFrame(list(zip(EpochID, EpochRisk, EpochSurv, EpochCensor)),
                        columns = ["Sample_ID", 'riskScore', 'Surv', 'Censorship'])
           
        sorted_surv_df = surv_df.sort_values(by='Sample_ID', ascending=False)
        
        return Epochloss, Epoch_CI, sorted_surv_df

def patient_test_surv_bin(model, test_loader):
    model.eval()
    grad_flag = False
    
    with torch.set_grad_enabled(grad_flag):
        EpochHazards = []
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        
        for index, mIHC_data in enumerate(test_loader,1):
                    
            tempsurvival = mIHC_data.survival 
            tempcensor = mIHC_data.censorship
            templabel = mIHC_data.surv_label

            mIHC_tempID = np.asarray(mIHC_data.item)
                        
            out_h = model(mIHC_data)
            hazards = torch.sigmoid(out_h)
            S = torch.cumprod(1 - hazards, dim=1)
            out_risk = -torch.sum(S, dim=1)
                          
            for hazardsval, riskval, survivalval, censorval in zip(hazards, out_risk,
                                tempsurvival, tempcensor):
                EpochHazards.append(hazardsval.cpu().detach())
                EpochRisk.append(riskval.cpu().detach().item())
                EpochSurv.append(survivalval.cpu().detach().item())
                EpochCensor.append(censorval.cpu().detach().item())
            
            for tempID in mIHC_tempID:
                 EpochID.append(tempID)
                 
        all_hazards_pred =  torch.cat(EpochHazards, dim=0).view(-1, hazards.shape[1])

        surv_df = pd.DataFrame(list(zip(EpochID, EpochRisk, EpochSurv, EpochCensor)),
                columns = ["Sample_ID", 'riskScore', 'Surv', 'Censorship'])
        
        hazards_columns = [f"H_score{i+1}" for i in range(all_hazards_pred.shape[1])]
        hazards_score_df = pd.DataFrame(all_hazards_pred, columns = hazards_columns)
   
        surv_hazards_stage_df = pd.concat([surv_df, hazards_score_df], axis=1)
        
        sorted_surv_df = surv_hazards_stage_df.sort_values(by='Sample_ID', ascending=False)
        sorted_surv_df['Patient_ID'] = sorted_surv_df['Sample_ID'].str.split('_', n=2).str[1]
    
        test_surv_result = sorted_surv_df.groupby('Patient_ID').agg({'riskScore': 'mean', 
                                            'Surv': 'first', 'Censorship': 'first'})
        
        test_CI = concordance_index_censored((1-test_surv_result['Censorship'].values).astype(bool), test_surv_result['Surv'].values,  test_surv_result['riskScore'].values, tied_tol=1e-08)[0]
           
        return test_CI, sorted_surv_df, test_surv_result
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss