import os
import timm
import torch
import numpy as np
import pandas as pd
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image

model = torch.load('Virchow_weights/Virchow_model.pth')
model.eval()
transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

patch_root_path = '../../preprocessed_data/HE_images/sub_HE_patches_filter'
features_root_path = '../../preprocessed_data/HE_images/sub_HE_patches_Virchow'
array_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']

for array_name in array_list:
    print(f'array_name:{array_name}')
    
    patch_array_path = os.path.join(patch_root_path, array_name)
    for patch_name in os.listdir(patch_array_path):
        print(f'patch_name:{patch_name}')
        patch_features_list = []
        sub_patch_path = os.path.join(patch_array_path, patch_name)

        sub_patch_image = Image.open(sub_patch_path)
        
        width, height = sub_patch_image.size

        block_width = width // 4
        block_height = height // 4

        sub_patch_feature_df = None
        for i in range(4):
            for j in range(4):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height

                block = sub_patch_image.crop((left, upper, right, lower))
                
                image = transforms(block).unsqueeze(0) 
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(image)

                class_token = output[:, 0]
                
                feature_np = class_token.numpy()
                feature_df = pd.DataFrame(feature_np, columns=range(class_token.shape[1]))
                
                if sub_patch_feature_df is None:
                    sub_patch_feature_df = feature_df
                else:
                    sub_patch_feature_df = pd.concat([sub_patch_feature_df, feature_df], ignore_index=True)
            
        sub_patch_mean_df = sub_patch_feature_df.mean().to_frame().T
                
        patch_features_path =os.path.join(features_root_path, array_name)
        if not os.path.exists(patch_features_path):
            os.makedirs(patch_features_path)
        
        patch_ID = patch_name.split('.')[0]
        sub_patch_mean_df.to_csv(os.path.join(patch_features_path, f'{patch_ID}.csv'), index=False)
