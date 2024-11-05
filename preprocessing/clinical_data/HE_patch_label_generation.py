import os
import torch
import pandas as pd

data_root_path = '../../preprocessed_data/clinical_data'
surv_csv_path = os.path.join(data_root_path, "surv_bin_clinical_data.csv")
surv_df = pd.read_csv(surv_csv_path)

HE_Virchow_path = '../../preprocessed_data/HE_images/sub_HE_patches_Virchow'

array_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']
counts_record_df = None
patches_dict = {}
for array_name in array_list:
    print("array_name:", array_name)
    
    HE_features_array = os.path.join(HE_Virchow_path, array_name)
    HE_features_files = os.listdir(HE_features_array)
    HE_features_names = [os.path.splitext(file)[0] for file in HE_features_files]
    print("len(HE_features_names):", len(HE_features_names))
    
    patches_dict[array_name] = HE_features_names
    
    counts = {}
    for item in HE_features_names:
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
for TMA_ID, HE_features_names in patches_dict.items():
    split_ID_df = pd.DataFrame([(x.split('_')[0], x) for x in HE_features_names], columns=['sample_ID', 'split_name'])
    split_ID_df['TMA_ID'] = TMA_ID
    if split_ID_all_df is None:
        split_ID_all_df = split_ID_df
    else:
        split_ID_all_df = pd.concat([split_ID_all_df, split_ID_df], ignore_index=True)
    
split_surv_df = merged_surv_df.merge(split_ID_all_df, on=['TMA_ID', 'sample_ID'], how='left')
split_surv_df['TMA_sample_ID'] = split_surv_df['TMA_ID'] + "+" +split_surv_df['sample_ID']

split_surv_df['sub_HE_features_path'] = HE_Virchow_path +"/"+ split_surv_df['TMA_ID'] +"/"+ split_surv_df['split_name']+'.csv'

#### For test
HE_features_test = pd.read_csv(split_surv_df['sub_HE_features_path'][0])

split_surv_df.to_csv(os.path.join(data_root_path, 'surv_bin_HE_Virchow_patches_df.csv'), index=False)