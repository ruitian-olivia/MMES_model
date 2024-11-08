import os
import sys
import torch
import argparse
from HE_ablation_surv_bin_train import Train

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")

    parser.add_argument("--model_name", default="HE_surv_bin", help="Model name", type=str)
    parser.add_argument("--GNN_type", default="HE1_graph", help="The type of GNN model", type=str)
    parser.add_argument("--TMA_test1", default="RJ_su03", help="Tissue microarray name as test set1", type=str)
    parser.add_argument("--TMA_test2", default="RJ_su09", help="Tissue microarray name as test set2", type=str)
    parser.add_argument("--surv_bins", default=4, help="The number of survival bins", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.0001, help="Weight decay rate", type=float)
    parser.add_argument("--epochs_num", default=10, help="Number of epochss", type=int)
    parser.add_argument("--dropout_rate", default=0.1, help="Dropout rate for MLP", type=float)
    parser.add_argument("--HE_nuclei_num", default=85, help="Number of node features in constructed HE Nuclei-Graph", type=int)
    parser.add_argument("--HE_features_num", default=1280, help="Number of extracted features for HE patch", type=int)
    parser.add_argument("--HE_dim_target", default=128, help="Number of embedding output for HE Nuclei-graph GNN", type=int)
    parser.add_argument("--batch_size", default=128 , help="batch size", type=int)
    parser.add_argument("--random_seed", default=10, help="The random seed", type=int)
    parser.add_argument("--layers",nargs='+',type=int,help="Dimension of the GNN hidden layer")
    parser.add_argument("--surv_mlp",nargs='+',type=int,help="Dimension of the MLP hidden layer in survival prediction task")

    Argument = parser.parse_args(args=[])

    return parser.parse_args()

def main():    
    Argument = Parser_main()

    Train(Argument)

if __name__ == "__main__":
    main()
