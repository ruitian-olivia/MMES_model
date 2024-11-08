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

class mIHC_hypergraph_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, mIHC_cell_num=None, mIHC_dim_target=128, layers=[128, 128, 128], surv_mlp=[64, 32], dropout = 0.25):
        super(mIHC_hypergraph_surv, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.surv_mlp = surv_mlp
        self.mIHC_convs = []
        self.mIHC_linears = []

        out_emb_global_dim = 0
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.mIHC_first_h = Sequential(Linear(mIHC_cell_num, out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.mIHC_convs.append(Sequential(HypergraphConv(input_emb_dim, out_emb_dim), ReLU()))
                
                out_emb_global_dim += out_emb_dim
                                
        self.mIHC_convs = torch.nn.ModuleList(self.mIHC_convs)
        
        self.mIHC_linears = Linear(out_emb_global_dim, mIHC_dim_target)
        
        risk_prediction_layers = []
                  
        input_dim = mIHC_dim_target
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

    def forward(self, mIHC_data=None):
        device = torch.device('cuda')
        
        x_mIHC, edge_index_mIHC, mIHC_batch = mIHC_data.x, mIHC_data.edge_index, mIHC_data.batch
        x_mIHC = x_mIHC.to(device)
        edge_index_mIHC = edge_index_mIHC.to(device)
        mIHC_batch = mIHC_batch.to(device)
        
        mIHC_global = None
        
        for layer in range(self.no_layers):
            if layer == 0:
                x_mIHC = self.mIHC_first_h(x_mIHC)
 
            else:
                
                x_mIHC = self.mIHC_convs[layer-1][0](x_mIHC, edge_index_mIHC)
                for GNN_layer in self.mIHC_convs[layer-1][1:]:
                    x_mIHC = GNN_layer(x_mIHC) 
                mIHC_out = global_mean_pool(x_mIHC, mIHC_batch)
                    
                if mIHC_global == None:
                    mIHC_global = mIHC_out
                else:
                    mIHC_global = torch.cat((mIHC_global, mIHC_out), 1)
                
        mIHC_out = self.mIHC_linears(mIHC_global)
    
        h = self.risk_prediction_layer(mIHC_out)
        
        return h

class mIHC_GCN_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, mIHC_cell_num=None, mIHC_dim_target=128, layers=[128, 128, 128], surv_mlp=[64, 32], dropout = 0.25):
        super(mIHC_GCN_surv, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.surv_mlp = surv_mlp
        self.mIHC_convs = []
        self.mIHC_linears = []
        
        self.mIHC_pool = []
        
        out_emb_global_dim = 0
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.mIHC_first_h = Sequential(Linear(mIHC_cell_num, out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.mIHC_convs.append(Sequential(GCNConv(input_emb_dim, out_emb_dim, add_self_loops=False), ReLU()))   
                self.mIHC_pool.append(SAGPooling(out_emb_dim, 0.6))
                
                out_emb_global_dim += out_emb_dim
                                
        self.mIHC_convs = torch.nn.ModuleList(self.mIHC_convs)
        self.mIHC_pool = torch.nn.ModuleList(self.mIHC_pool) 
        self.mIHC_linears = Linear(out_emb_global_dim, mIHC_dim_target)

        risk_prediction_layers = []
        
        input_dim = mIHC_dim_target
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
        

    def forward(self, mIHC_data=None):
        device = torch.device('cuda')
        
        x_mIHC, edge_index_mIHC, mIHC_batch = mIHC_data.x, mIHC_data.edge_index, mIHC_data.batch
        x_mIHC = x_mIHC.to(device)
        edge_index_mIHC = edge_index_mIHC.to(device)
        mIHC_batch = mIHC_batch.to(device)
        
        mIHC_global = None
        
        for layer in range(self.no_layers):
            # print("layer:", layer)         
            if layer == 0:
                x_mIHC = self.mIHC_first_h(x_mIHC)
 
            else:
                
                x_mIHC = self.mIHC_convs[layer-1][0](x_mIHC, edge_index_mIHC)
                for GNN_layer in self.mIHC_convs[layer-1][1:]:
                    x_mIHC = GNN_layer(x_mIHC) 
                x_mIHC,edge_index_mIHC,_,mIHC_batch,_,_ = self.mIHC_pool[layer-1](
                    x=x_mIHC,
                    edge_index=edge_index_mIHC,
                    batch=mIHC_batch
                )
                mIHC_out = global_mean_pool(x_mIHC, mIHC_batch)
                    
                if mIHC_global == None:
                    mIHC_global = mIHC_out
                else:
                    mIHC_global = torch.cat((mIHC_global, mIHC_out), 1)
        
        mIHC_out = self.mIHC_linears(mIHC_global)
                    
        h = self.risk_prediction_layer(mIHC_out)
        
        return h

#  self.mIHC_convs.append(Sequential(GATConv(input_emb_dim, out_emb_dim, add_self_loops=False), ReLU()))

class mIHC_GAT_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, mIHC_cell_num=None, mIHC_dim_target=128, layers=[128, 128, 128], surv_mlp=[64, 32], dropout = 0.25):
        super(mIHC_GAT_surv, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.surv_mlp = surv_mlp
        self.mIHC_convs = []
        self.mIHC_linears = []
        
        self.mIHC_pool = []
        
        out_emb_global_dim = 0
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.mIHC_first_h = Sequential(Linear(mIHC_cell_num, out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.mIHC_convs.append(Sequential(GATConv(input_emb_dim, out_emb_dim, add_self_loops=False), ReLU()))
                self.mIHC_pool.append(SAGPooling(out_emb_dim, 0.6))
                
                out_emb_global_dim += out_emb_dim
                                
        self.mIHC_convs = torch.nn.ModuleList(self.mIHC_convs)
        self.mIHC_pool = torch.nn.ModuleList(self.mIHC_pool) 
        self.mIHC_linears = Linear(out_emb_global_dim, mIHC_dim_target)

        risk_prediction_layers = []
        
        input_dim = mIHC_dim_target
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
        

    def forward(self, mIHC_data=None):
        device = torch.device('cuda')
        
        x_mIHC, edge_index_mIHC, mIHC_batch = mIHC_data.x, mIHC_data.edge_index, mIHC_data.batch
        x_mIHC = x_mIHC.to(device)
        edge_index_mIHC = edge_index_mIHC.to(device)
        mIHC_batch = mIHC_batch.to(device)
        
        mIHC_global = None
        
        for layer in range(self.no_layers):
            # print("layer:", layer)         
            if layer == 0:
                x_mIHC = self.mIHC_first_h(x_mIHC)
 
            else:
                
                x_mIHC = self.mIHC_convs[layer-1][0](x_mIHC, edge_index_mIHC)
                for GNN_layer in self.mIHC_convs[layer-1][1:]:
                    x_mIHC = GNN_layer(x_mIHC) 
                x_mIHC,edge_index_mIHC,_,mIHC_batch,_,_ = self.mIHC_pool[layer-1](
                    x=x_mIHC,
                    edge_index=edge_index_mIHC,
                    batch=mIHC_batch
                )
                mIHC_out = global_mean_pool(x_mIHC, mIHC_batch)
                    
                if mIHC_global == None:
                    mIHC_global = mIHC_out
                else:
                    mIHC_global = torch.cat((mIHC_global, mIHC_out), 1)
        
        mIHC_out = self.mIHC_linears(mIHC_global)
                    
        h = self.risk_prediction_layer(mIHC_out)
        
        return h

class mIHC_GIN_surv(torch.nn.Module):
    def __init__(self, surv_bins=4, mIHC_cell_num=None, mIHC_dim_target=128, layers=[128, 128, 128], surv_mlp=[64, 32], dropout = 0.25):
        super(mIHC_GIN_surv, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.surv_mlp = surv_mlp
        self.mIHC_convs = []
        self.mIHC_linears = []
        
        self.mIHC_pool = []
        
        out_emb_global_dim = 0
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.mIHC_first_h = Sequential(Linear(mIHC_cell_num, out_emb_dim), ReLU())
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                nn_module = nn.Sequential(
                                    nn.Linear(input_emb_dim, input_emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(input_emb_dim, out_emb_dim)
                                )
                self.mIHC_convs.append(GINConv(nn_module))
                self.mIHC_pool.append(SAGPooling(out_emb_dim, 0.6))
                
                out_emb_global_dim += out_emb_dim
                                
        self.mIHC_convs = torch.nn.ModuleList(self.mIHC_convs)
        self.mIHC_pool = torch.nn.ModuleList(self.mIHC_pool) 
        self.mIHC_linears = Linear(out_emb_global_dim, mIHC_dim_target)

        risk_prediction_layers = []
        
        input_dim = mIHC_dim_target
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
        
    def forward(self, mIHC_data=None):
        device = torch.device('cuda')
        
        x_mIHC, edge_index_mIHC, mIHC_batch = mIHC_data.x, mIHC_data.edge_index, mIHC_data.batch
        x_mIHC = x_mIHC.to(device)
        edge_index_mIHC = edge_index_mIHC.to(device)
        mIHC_batch = mIHC_batch.to(device)
        
        mIHC_global = None
        
        for layer in range(self.no_layers):
            # print("layer:", layer)         
            if layer == 0:
                x_mIHC = self.mIHC_first_h(x_mIHC)
 
            else:
                x_mIHC = self.mIHC_convs[layer-1](x_mIHC, edge_index_mIHC)
                x_mIHC,edge_index_mIHC,_,mIHC_batch,_,_ = self.mIHC_pool[layer-1](
                    x=x_mIHC,
                    edge_index=edge_index_mIHC,
                    batch=mIHC_batch
                )
                mIHC_out = global_mean_pool(x_mIHC, mIHC_batch)
                    
                if mIHC_global == None:
                    mIHC_global = mIHC_out
                else:
                    mIHC_global = torch.cat((mIHC_global, mIHC_out), 1)
        
        mIHC_out = self.mIHC_linears(mIHC_global)
                    
        h = self.risk_prediction_layer(mIHC_out)
        
        return h
