import os
import torch
import pandas as pd

data_root_path = '../../preprocessed_data/clinical_data'
surv_csv_path = os.path.join(data_root_path, "surv_bin_clinical_data.csv")
surv_df = pd.read_csv(surv_csv_path)

mIHC_graph_path = '../../preprocessed_data/mIHC_data/constructed_mIHC_graph_split'

array_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']
counts_record_df = None
patches_dict = {}
for array_name in array_list:
    print("array_name:", array_name)
    
    mIHC_graph_array = os.path.join(mIHC_graph_path, array_name)
    mIHC_graph_files = os.listdir(mIHC_graph_array)
    mIHC_graph_names = [os.path.splitext(file)[0] for file in mIHC_graph_files]
    print("len(mIHC_graph_names):", len(mIHC_graph_names))
    
    patches_dict[array_name] = mIHC_graph_names
    
    counts = {}
    for item in mIHC_graph_names:
        key = item.split('_')[0]
        counts[key] = counts.get(key, 0) + 1

    counts_df = pd.DataFrame(list(counts.items()), columns=['sample_ID', 'Split_counts'])
    counts_df['TMA_ID'] = TMA_ID
    
    if counts_record_df is None:
        counts_record_df = counts_df
    else:
        counts_record_df = pd.concat([counts_record_df, counts_df], ignore_index=True)
    
merged_surv_df = surv_df.merge(counts_record_df, on=['TMA_ID', 'sample_ID'], how='left')

## Generate new dataframe
split_ID_all_df = None
for TMA_ID, mIHC_graph_names in patches_dict.items():
    split_ID_df = pd.DataFrame([(x.split('_')[0], x) for x in mIHC_graph_names], columns=['sample_ID', 'split_name'])
    split_ID_df['TMA_ID'] = TMA_ID
    if split_ID_all_df is None:
        split_ID_all_df = split_ID_df
    else:
        split_ID_all_df = pd.concat([split_ID_all_df, split_ID_df], ignore_index=True)

split_surv_df = merged_surv_df.merge(split_ID_all_df, on=['TMA_ID', 'sample_ID'], how='left')
split_surv_df['TMA_sample_ID'] = split_surv_df['TMA_ID'] + "+" +split_surv_df['sample_ID']

split_surv_df['sub_mIHC_graph_path'] = mIHC_graph_path +"/"+ split_surv_df['TMA_ID'] +"/"+ split_surv_df['split_name']+'.pt'

#### For test
mIHC_graph_test = torch.load(split_surv_df['sub_mIHC_graph_path'][0])

split_surv_df.to_csv(os.path.join(data_root_path, 'surv_bin_mIHC_patches_df.csv'), index=False)