import os
import cv2
import skimage.io
import skimage.measure
import skimage.color
import openslide
import shutil
import numpy as np
import histomicstk as htk
import wsi_tile_cleanup as cleanup
from PIL import Image

array_root_path = '../../preprocessed_data/HE_images/TMA_dearray/'
array_list = ['TMA1', 'TMA2', 'TMA3', 'TMA4', 'TMA5']

ref_HE_path = 'L1.png'
im_reference = skimage.io.imread(ref_HE_path)[:, :, :3]
# get mean and stddev of reference image in lab space
mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

patches_root_path = '../../preprocessed_data/HE_images/sub_HE_patches'
patches_filter_root_path = '../../preprocessed_data/HE_images/sub_HE_patches_filter'

for array_name in array_list:
    print("array_name:", array_name)
    array_path = os.path.join(array_root_path, array_name)
    
    sample_list = os.listdir(array_path)
    sample_list.sort()
    print("len(sample_list):", len(sample_list))
    for sample_name in sample_list:
        print("Sample name:", sample_name)
        sample_array_path = os.path.join(array_path, sample_name)
        
        sample_ID = sample_name.split('.')[0]
        print("Sample ID:", sample_ID)
        patches_path = os.path.join(patches_root_path, array_name, sample_ID)
        patches_other_path = os.path.join(patches_root_path, array_name+"_filtered", sample_ID)
        patches_filter_path = os.path.join(patches_filter_root_path, array_name)
        if not os.path.isdir(patches_path):
            os.makedirs(patches_path)
        if not os.path.isdir(patches_other_path):
            os.makedirs(patches_other_path)
        if not os.path.isdir(patches_filter_path):
            os.makedirs(patches_filter_path)

        vi_nmzd = cleanup.utils.read_image(sample_array_path)
        otsu = cleanup.filters.otsu_threshold(vi_nmzd)
        print(f"otsu_threshold: {otsu}")
        
        im_input = Image.open(sample_array_path)
        width, height = im_input.size

        assert width == height, "width and height are not equal"

        num_patches = 4
        patch_size = 3200/num_patches
        print("patch_size:", patch_size)
        
        for y in range(num_patches):
            for x in range(num_patches):
                left = x * patch_size
                top = y * patch_size
                patch = im_input.crop((left, top, left + patch_size, top + patch_size))

                patch.save(os.path.join(patches_path, f"patch_x{x+1}_y{y+1}.png"))
    
        print("Filtering patches begin!")   
        for filename in os.listdir(patches_path):
            if filename.endswith('png'):
                try:
                    print("filename:", filename)
                    print("saved png name:", sample_ID+filename[6:])
                    tile_path = os.path.join(patches_path, filename)

                    vi_tile = cleanup.utils.read_image(tile_path)
                    
                    bands = cleanup.utils.split_rgb(vi_tile)

                    perc_bg = cleanup.filters.bg_percent(bands)
                    print(f"bg_percent: {perc_bg*100:.3f}%")

                    if perc_bg < 0.85: # threshold for background percentage
                        im_input = skimage.io.imread(tile_path)[:, :, :3]

                        # perform reinhard color normalization
                        im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)
                        pil_img = Image.fromarray(im_nmzd)
                        pil_img.save(os.path.join(patches_filter_path, sample_ID+filename[5:]))
                        
                    else:
                        shutil.copy(tile_path, os.path.join(patches_other_path, filename))
                        
                except:
                    print("Error occured in patch: %s" % os.path.join(patches_path, filename))
                    