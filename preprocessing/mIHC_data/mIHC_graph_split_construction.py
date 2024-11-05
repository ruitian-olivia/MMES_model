import os
import re
import glob
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform

preprocessed_mIHC_csv_path = '../../preprocessed_data/mIHC_data/preprocessed_mIHC_csv'
constructed_mIHC_graph_path = '../../preprocessed_data/mIHC_data/constructed_mIHC_graph_split'

TMA_name_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']

# critical distance 20 \mu m, pix width 0.5 \mu m
pix = 0.5
critical = 20/pix

for TMA_name in TMA_name_list:
    print("TMA_name:", TMA_name)
    
    TMA_csv_path = os.path.join(preprocessed_mIHC_csv_path, TMA_name)
    TMA_graph_path = os.path.join(constructed_mIHC_graph_path, TMA_name)

    if not os.path.exists(TMA_graph_path): 
        os.makedirs(TMA_graph_path)
        
    csv_files = glob.glob(TMA_csv_path + '/*.csv')
        
    for csv_file_path in csv_files:
        patient_ID = os.path.splitext(os.path.basename(csv_file_path))[0].split("_")[0]
        print("patient_ID:", patient_ID)

        nuclei_nmzd_feature_path = os.path.join(csv_file_path)
        nuclei_nmzd_feature_df = pd.read_csv(nuclei_nmzd_feature_path)
        
        split_k = 4
        
        nuclei_nmzd_feature_df['X_label'] = pd.cut(nuclei_nmzd_feature_df['X_coor'], split_k, labels=range(split_k))
        nuclei_nmzd_feature_df['Y_label'] = pd.cut(nuclei_nmzd_feature_df['Y_coor'], split_k, labels=range(split_k))
        nuclei_nmzd_feature_groups = nuclei_nmzd_feature_df.groupby(['X_label', 'Y_label'])
        
        sub_graph_count = 0
        for x_label in range(split_k):
            for y_label in range(split_k):
                try:
                    mIHC_sub_df = nuclei_nmzd_feature_groups.get_group((x_label, y_label))
                    num_cell = len(mIHC_sub_df)
                    
                    if num_cell >= 100:
                        sub_graph_count += 1
                        
                        feature_np = np.array(mIHC_sub_df.drop(['X_label', 'Y_label', 'X_coor', 'Y_coor'], axis=1))
                        feature_tensor = torch.tensor(feature_np, dtype=torch.float)

                        pos_np = np.array(mIHC_sub_df[['X_coor', 'Y_coor']])
                        pos_tensor = torch.tensor(pos_np, dtype=torch.float)
                        
                        coordinates = pos_np
                        distance_matrix = pdist(coordinates, metric='euclidean')
                        distance_matrix = squareform(distance_matrix)
                        distance_df = pd.DataFrame(distance_matrix, columns=mIHC_sub_df.index, index=mIHC_sub_df.index)

                        incidence_matrix = (distance_df <= critical) * np.ones(distance_df.shape)

                        incidence_matrix = torch.Tensor(incidence_matrix.to_numpy())
                        edge_index = incidence_matrix.nonzero().t().contiguous()
                        x = torch.tensor(edge_index)
                        index = torch.LongTensor([1, 0])
                        y = torch.zeros_like(x)
                        y[index] = x
                        edge_index = y

                        graph_data = Data(x=feature_tensor, edge_index=edge_index, pos=pos_np)
                        graph_name = patient_ID+'_x'+str(x_label+1)+'_y'+str(y_label+1)+'.pt'
                        print("graph_data:", graph_data)

                        torch.save(graph_data, os.path.join(TMA_graph_path, graph_name))
                except:
                    continue
                                    