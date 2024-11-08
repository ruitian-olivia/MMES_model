import os
import cv2
import json
import glob
import torch
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv, check_y_survival
import lifelines.utils.concordance as LUC
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

def getGender(gender):
    if gender == "male":
        return 1
    elif gender == "female":
        return 0
    
def getGrade(grade):
    if grade == "low":
        return "1"
    elif grade == "middle":
        return "2"
    elif grade == "high":
        return "3"

def merge_surv_df(surv_df, HE_surv_df, mIHC_surv_df):
    
    surv_df['Patient_ID'] = surv_df['TMA_ID'].str.split('_').str[1] + "+" + surv_df['sample_ID']
    HE_surv_df['Patient_ID'] = HE_surv_df['Sample_ID'].str.split('_', n=2).str[1]
    mIHC_surv_df['Patient_ID'] = mIHC_surv_df['Sample_ID'].str.split('_', n=2).str[1]
    
    HE_surv_result = HE_surv_df.groupby('Patient_ID').agg({
    'H_score1': 'mean',
    'H_score2': 'mean',
    'H_score3': 'mean',
    'H_score4': 'mean', 
    'Surv': 'first', 'Censorship': 'first'})
    
    new_HE_surv_result = pd.DataFrame(HE_surv_result[['H_score1', 'H_score2', 'H_score3', 'H_score4']])
    new_HE_surv_result = new_HE_surv_result.rename(columns={'H_score1': 'HE_H_score1', 
                                                           'H_score2': 'HE_H_score2',
                                                           'H_score3': 'HE_H_score3',
                                                           'H_score4': 'HE_H_score4'})
    
    mIHC_surv_result = mIHC_surv_df.groupby('Patient_ID').agg({
    'H_score1': 'mean',
    'H_score2': 'mean',
    'H_score3': 'mean',
    'H_score4': 'mean', 
    'Surv': 'first', 'Censorship': 'first'})
    
    new_mIHC_surv_result = pd.DataFrame(mIHC_surv_result[['H_score1', 'H_score2', 'H_score3', 'H_score4']])
    new_mIHC_surv_result = new_mIHC_surv_result.rename(columns={'H_score1': 'mIHC_H_score1', 
                                                           'H_score2': 'mIHC_H_score2',
                                                           'H_score3': 'mIHC_H_score3',
                                                           'H_score4': 'mIHC_H_score4'})
    
    merged_surv_result = new_HE_surv_result.merge(new_mIHC_surv_result, left_index=True, right_index=True, how='inner')
    
    surv_cli_df =  merged_surv_result.merge(surv_df, left_index=True, right_on='Patient_ID')
    
    surv_cli_df['gender'] = surv_cli_df.apply(lambda x: getGender(x.gender), axis = 1)
    surv_cli_df['grade'] = surv_cli_df.apply(lambda x: getGrade(x.grade), axis = 1)
    dummy_df = pd.get_dummies(surv_cli_df['grade'], prefix='grade')
    surv_cli_df = pd.concat([surv_cli_df, dummy_df], axis=1)
    
    return surv_cli_df

if not os.path.exists('saved_weights'):
    os.makedirs('saved_weights')
    
if not os.path.exists('saved_results'):
    os.makedirs('saved_results')
    
surv_csv_path = '../../preprocessed_data/clinical_data/surv_bin_clinical_data.csv'
surv_df = pd.read_csv(surv_csv_path)
    
HE_df_root = '../HE_modality/model_results/HE_features_Virchow_surv_bin/surv_record'
HE_train_surv_df = pd.read_csv(os.path.join(HE_df_root, 'train_surv_df.csv'))
HE_test1_surv_df = pd.read_csv(os.path.join(HE_df_root, 'test1_surv_df.csv'))
HE_test2_surv_df = pd.read_csv(os.path.join(HE_df_root, 'test2_surv_df.csv'))

mIHC_df_root = '../mIHC_modality/model_results/hyper_mIHC_surv_bin/surv_record'
mIHC_train_surv_df = pd.read_csv(os.path.join(mIHC_df_root, 'train_surv_df.csv'))
mIHC_test1_surv_df = pd.read_csv(os.path.join(mIHC_df_root, 'test1_surv_df.csv'))
mIHC_test2_surv_df = pd.read_csv(os.path.join(mIHC_df_root, 'test2_surv_df.csv'))

train_surv_cli_df = merge_surv_df(surv_df, HE_train_surv_df, mIHC_train_surv_df)
train_surv_cli_df['gender'] = train_surv_cli_df['gender'].fillna(train_surv_cli_df['gender'].mode()[0])
train_surv_cli_df['age'] = train_surv_cli_df['age'].fillna(train_surv_cli_df['age'].mean())

feature_cols = ['HE_H_score1', 'HE_H_score2', 'HE_H_score3', 'HE_H_score4', 'mIHC_H_score1', 'mIHC_H_score2', 'mIHC_H_score3', 'mIHC_H_score4', 'gender', 'age', 'grade_2', 'grade_3', 'surv_time', 'event']

disc_cph_marker = CoxPHFitter()
disc_X = train_surv_cli_df[feature_cols]
disc_cph_marker.fit(disc_X, duration_col='surv_time', event_col='event')
print(disc_cph_marker.summary)

with open("./saved_weights/MMES.pkl", "wb") as file:
    pickle.dump(disc_cph_marker, file)

# For Test set 1
test1_surv_cli_df = merge_surv_df(surv_df, HE_test1_surv_df, mIHC_test1_surv_df)
test1_X = test1_surv_cli_df[feature_cols]
test1_risk_scores = disc_cph_marker.predict_partial_hazard(test1_X)
test1_X['fusion_score'] = test1_risk_scores
test1_X.to_csv('./saved_results/MMES_test1_X.csv', index=False)

test1_c_index = disc_cph_marker.score(test1_X, scoring_method="concordance_index")
print(f'Test1 C-index: {test1_c_index:.4f}')

# Bootstrap for C-index 95% CI
n_bootstraps = 1000
c_index_bootstraps = []

for i in range(n_bootstraps):
    bootstrap_sample = resample(test1_X, replace=True, random_state=i)
    c_index_bootstrap = disc_cph_marker.score(bootstrap_sample, scoring_method="concordance_index")
    c_index_bootstraps.append(c_index_bootstrap)

alpha = 0.05
lower = np.percentile(c_index_bootstraps, 100 * alpha / 2)
upper = np.percentile(c_index_bootstraps, 100 * (1 - alpha / 2))
print(f'95% CI for C-index: ({lower:.3f}, {upper:.3f})')

disc_X_structured = Surv.from_dataframe('event', 'surv_time', disc_X)
test1_X_structured = Surv.from_dataframe('event', 'surv_time', test1_X)
times = [12, 24, 36, 48, 60]
test1_aucs, test1_mean_auc = cumulative_dynamic_auc(disc_X_structured, test1_X_structured, test1_X['fusion_score'], times)
print(f"Mean time-dependent AUC: {test1_mean_auc: .3f}")

survival_function = disc_cph_marker.predict_survival_function(test1_X, times=times)
survival_function_np = survival_function.to_numpy().T
test1_ibs = integrated_brier_score(test1_X_structured, test1_X_structured, survival_function_np, times)
print(f"Integrated Brier Score: {test1_ibs: .3f}")

# For Test set 2
test2_surv_cli_df = merge_surv_df(surv_df, HE_test2_surv_df, mIHC_test2_surv_df)
test2_X = test2_surv_cli_df[feature_cols]
test2_risk_scores = disc_cph_marker.predict_partial_hazard(test2_X)
test2_X['fusion_score'] = test2_risk_scores
test2_X.to_csv('./saved_results/MMES_test2_X.csv', index=False)

test2_c_index = disc_cph_marker.score(test2_X, scoring_method="concordance_index")
print(f'Test2 C-index: {test2_c_index:.3f}')

n_bootstraps = 1000
c_index_bootstraps = []
for i in range(n_bootstraps):
    bootstrap_sample = resample(test2_X, replace=True, random_state=i)
    c_index_bootstrap = disc_cph_marker.score(bootstrap_sample, scoring_method="concordance_index")
    c_index_bootstraps.append(c_index_bootstrap)

alpha = 0.05
lower = np.percentile(c_index_bootstraps, 100 * alpha / 2)
upper = np.percentile(c_index_bootstraps, 100 * (1 - alpha / 2))
print(f'95% CI for C-index: ({lower:.3f}, {upper:.3f})')

disc_X_structured = Surv.from_dataframe('event', 'surv_time', disc_X)
test2_X_structured = Surv.from_dataframe('event', 'surv_time', test2_X)
times = [12, 24, 36, 48, 60]
test2_aucs, test2_mean_auc = cumulative_dynamic_auc(disc_X_structured, test2_X_structured, test2_X['fusion_score'], times)
print(f"Mean time-dependent AUC: {test2_mean_auc: .3f}")

survival_function = disc_cph_marker.predict_survival_function(test2_X, times=times)
survival_function_np = survival_function.to_numpy().T
test2_ibs = integrated_brier_score(test2_X_structured, test2_X_structured, survival_function_np, times)
print(f"Integrated Brier Score: {test2_ibs: .3f}")

# For all-test cohort
test_X_all = pd.concat([test1_X, test2_X])
test_risk_scores = disc_cph_marker.predict_partial_hazard(test_X_all)
test_X_all['fusion_score'] = test_risk_scores
test_X_all.to_csv('./saved_results/MMES_all_test_X.csv', index=False)

test_c_index = disc_cph_marker.score(test_X_all, scoring_method="concordance_index")
print(f'All test C-index: {test_c_index:.3f}')

n_bootstraps = 1000
c_index_bootstraps = []
for i in range(n_bootstraps):
    bootstrap_sample = resample(test_X_all, replace=True, random_state=i)
    c_index_bootstrap = disc_cph_marker.score(bootstrap_sample, scoring_method="concordance_index")
    c_index_bootstraps.append(c_index_bootstrap)
alpha = 0.05
lower = np.percentile(c_index_bootstraps, 100 * alpha / 2)
upper = np.percentile(c_index_bootstraps, 100 * (1 - alpha / 2))
print(f'95% CI for C-index: ({lower:.3f}, {upper:.3f})')

disc_X_structured = Surv.from_dataframe('event', 'surv_time', disc_X)
test_X_all_structured = Surv.from_dataframe('event', 'surv_time', test_X_all)
times = [12, 24, 36, 48, 60]
aucs, mean_auc = cumulative_dynamic_auc(disc_X_structured, test_X_all_structured, test_X_all['fusion_score'], times)
print(f"Mean time-dependent AUC: {mean_auc: .3f}")

survival_function = disc_cph_marker.predict_survival_function(test_X_all, times=times)
survival_function_np = survival_function.to_numpy().T
ibs = integrated_brier_score(test_X_all_structured, test_X_all_structured, survival_function_np, times)
print(f"Integrated Brier Score: {ibs: .3f}")