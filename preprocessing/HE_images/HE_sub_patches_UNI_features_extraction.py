import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

model_weights = './UNI_weights/pytorch_model.bin'

model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)

model.load_state_dict(torch.load(model_weights, map_location="cpu"), strict=True)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()

patch_root_path = '../../preprocessed_data/HE_images/sub_HE_patches_filter'
features_root_path = '../../preprocessed_data/HE_images/sub_HE_patches_UNI'
array_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']

for array_name in array_list:
    print(f'array_name:{array_name}')
    
    patch_array_path = os.path.join(patch_root_path, array_name)
    for patch_name in os.listdir(patch_array_path):
        print(f'patch_name:{patch_name}')
        patch_features_list = []
        sub_patch_path = os.path.join(patch_array_path, patch_name)

        sub_patch_image = Image.open(sub_patch_path)
        sub_patch_image = transform(sub_patch_image).unsqueeze(dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
        with torch.inference_mode():
            feature_emb = model(sub_patch_image) # Extracted features (torch.Tensor) with shape [1,1024]
        feature_np = feature_emb.numpy()
        feature_df = pd.DataFrame(feature_np, columns=range(1024))

        patch_features_path =os.path.join(features_root_path, array_name)
        if not os.path.exists(patch_features_path):
            os.makedirs(patch_features_path)
        
        patch_ID = patch_name.split('.')[0]
        feature_df.to_csv(os.path.join(patch_features_path, f'{patch_ID}.csv'), index=False)
                