import os
import numpy as np
import pandas as pd

data_root_path = '../../preprocessed_data/clinical_data'
surv_csv_path = os.path.join(data_root_path, 'raw_clinical_data.csv')
surv_df = pd.read_csv(surv_csv_path)

def getCensor(status):
    if status == 'death':
        return 0
    elif status == 'censored':
        return 1

surv_df['censorship'] = surv_df.apply(lambda x: getCensor(x.status), axis = 1)

uncensored_df = surv_df[surv_df['censorship'] < 1]

n_bins = 4
disc_labels, q_bins = pd.qcut(uncensored_df['surv_time'], q=n_bins, retbins=True, labels=False)

eps=1e-6
q_bins[-1] = surv_df['surv_time'].max() + eps
q_bins[0] = surv_df['surv_time'].min() - eps

disc_labels, q_bins = pd.cut(surv_df['surv_time'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
surv_df.insert(2, 'surv_label', disc_labels.values.astype(int))

surv_df.to_csv(os.path.join(data_root_path, "surv_bin_clinical_data.csv"), index=False)