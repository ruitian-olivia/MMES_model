import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch.nn import Sequential
from torch.nn import Linear, Bilinear
from torch.nn import ReLU, Sigmoid, Softmax
from torch_geometric.nn import global_mean_pool, AttentionalAggregation
from torch_geometric.nn import HypergraphConv, GCNConv, GATConv, GINConv
from torch_geometric.nn.pool import SAGPooling

from model_utils import Attn_Net_Gated, MultiheadAttention

class HE_graph_hypergraph_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, HE_nuclei_num=None, HE_dim_target=128, layers=[128, 128, 128], surv_mlp=[64, 32], dropout = 0.25):
        super(HE_graph_hypergraph_surv, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.surv_mlp = surv_mlp
        self.HE_nuclei_convs = []
        self.HE_nuclei_linears = []

        out_emb_global_dim = 0
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.HE_nuclei_first_h = Sequential(Linear(HE_nuclei_num, out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.HE_nuclei_convs.append(Sequential(HypergraphConv(input_emb_dim, out_emb_dim), ReLU()))
                
                out_emb_global_dim += out_emb_dim
                                
        self.HE_nuclei_convs = torch.nn.ModuleList(self.HE_nuclei_convs)
        self.HE_linears = Linear(out_emb_global_dim, HE_dim_target)
        
        risk_prediction_layers = []
                  
        input_dim = HE_dim_target
        for hidden_dim in self.surv_mlp:
            if hidden_dim < len(self.surv_mlp) - 1:
                risk_prediction_layers.append(Linear(input_dim, hidden_dim))
                risk_prediction_layers.append(ReLU())
                risk_prediction_layers.append(Dropout(p=dropout))
                input_dim = hidden_dim
            else:
                risk_prediction_layers.append(Linear(input_dim, hidden_dim))
                risk_prediction_layers.append(BatchNorm1d(hidden_dim)) 
                risk_prediction_layers.append(ReLU())
                risk_prediction_layers.append(Dropout(p=dropout))
                input_dim = hidden_dim
        risk_prediction_layers.append(Linear(input_dim, surv_bins, bias=False))
        self.risk_prediction_layer = Sequential(*risk_prediction_layers)

    def forward(self, HE_data=None):
        device = torch.device('cuda')
        
        x_HE_nuclei, edge_index_HE_nuclei, HE_nuclei_batch = HE_data.x, HE_data.edge_index, HE_data.batch
        x_HE_nuclei = x_HE_nuclei.to(device)
        edge_index_HE_nuclei = edge_index_HE_nuclei.to(device)
        HE_nuclei_batch = HE_nuclei_batch.to(device)
        
        HE_features = HE_data.HE_features.to(device)
        
        HE_nuclei_global = None
        
        for layer in range(self.no_layers):
            if layer == 0:
                x_HE_nuclei = self.HE_nuclei_first_h(x_HE_nuclei)
 
            else:
                x_HE_nuclei = self.HE_nuclei_convs[layer-1][0](x_HE_nuclei, edge_index_HE_nuclei)
                for GNN_layer in self.HE_nuclei_convs[layer-1][1:]:
                    x_HE_nuclei = GNN_layer(x_HE_nuclei) 
                HE_nuclei_out = global_mean_pool(x_HE_nuclei, HE_nuclei_batch)

                if HE_nuclei_global == None:
                    HE_nuclei_global = HE_nuclei_out
                else:
                    HE_nuclei_global = torch.cat((HE_nuclei_global, HE_nuclei_out), 1)
                
        HE_out = self.HE_linears(HE_nuclei_global)
    
        h = self.risk_prediction_layer(HE_out)
        
        return h

class HE_features_hypergraph_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, HE_features_num=None, HE_dim_target=128, surv_mlp=[64, 32], dropout = 0.25):
        super(HE_features_hypergraph_surv, self).__init__()
        self.dropout = dropout
        self.surv_mlp = surv_mlp
                                
        self.HE_linears = Linear(HE_features_num, HE_dim_target)

        risk_prediction_layers = []
                  
        input_dim = HE_dim_target
        for hidden_dim in self.surv_mlp:
            if hidden_dim < len(self.surv_mlp) - 1:
                risk_prediction_layers.append(Linear(input_dim, hidden_dim))
                risk_prediction_layers.append(ReLU())
                risk_prediction_layers.append(Dropout(p=dropout))
                input_dim = hidden_dim
            else:
                risk_prediction_layers.append(Linear(input_dim, hidden_dim))
                risk_prediction_layers.append(BatchNorm1d(hidden_dim)) 
                risk_prediction_layers.append(ReLU())
                risk_prediction_layers.append(Dropout(p=dropout))
                input_dim = hidden_dim
        risk_prediction_layers.append(Linear(input_dim, surv_bins, bias=False))
        self.risk_prediction_layer = Sequential(*risk_prediction_layers)

    def forward(self, HE_data=None):
        device = torch.device('cuda')
        
        HE_features = HE_data.HE_features.to(device)
                
        HE_out = self.HE_linears(HE_features)
    
        h = self.risk_prediction_layer(HE_out)
        
        return h
