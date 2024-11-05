## MMES_model

### I. Data preprocessing

Code in **./preprocessing**

##### 1. HE-stained images preprocessing
In **./preprocessing/HE_images** directory

###### 1.1 HE-stained tissue microarray (TMA) de-array
For HE-stained TMAs slides scanned by a digital microscope under the magnification of 40×, we decomposed TMAs into 3200×3200 pixel HE-stained images corresponding to each sample core, with the resolution of 0.5 µm/pixel. 

TMA dearrayer tutorial: [De-array a Tissue Microarray (TMA) using QuPath and Python](https://andrewjanowczyk.com/de-array-a-tissue-microarray-(tma)-using-qupath-and-python/)


```bash
python tma_spot_dearray.py '../../preprocessed_data/HE_images/HE_TMA/TMA1.svs' '../../preprocessed_data/HE_images/HE_TMA/TMA1.txt' -s 3200 -o '../../preprocessed_data/HE_images/TMA_dearray/TMA1'
```

For the script tma_spot_dearray.py, the first argument represents the TMA .svs file path, and the second argument represents the TMA spot location .txt file (outputted by [Qupath software](https://qupath.github.io/)) path. '-s' argument represents TMA spot size, and '-o' argument represents target output directory.

###### 1.2 HE patches splitting and normalization
We then segmented each core images into non-overlapping 800×800 pixel patches. We considered pixels with intensity values greater than 200 in all three RGB channels as background, and filtered out patches where the background area is 85% or more. We performed Reinhard color normalization for all HE-stained patches.

```bash
python HE_sub_patches_extraction.py
```

###### 1.3 HE features extraction
We used foundation models in the field of pathology, including Virchow<sup>[1]</sup> and UNI<sup>[2]</sup>, to extract features from HE-stained images.

For Virchow, with an input image size of 224×224, we divided and resized our 800×800 pixel patches into 16 tiles with 224×224 pixel as inputs to the foundation model. We then averaged the resulting 16 sets of 1280-dimensional features to obtain patch-level 1280-dimensional embedding features. We downloaded the model weights of [Virchow](https://huggingface.co/paige-ai/Virchow) from Hugging Face, which saved in **./Virchow_weights/Virchow_model.pth**. 

```bash
python HE_sub_patches_Virchow_features_extraction.py
```
Extracted Virchow features are saved in **../../preprocessed_data/HE_images/sub_HE_patches_Virchow**. 

For UNI, which demonstrates robustness to different image resolutions, we directly resized each 800×800 pixel patch to 224×224 as the input for the foundation model, obtaining 1024-dimensional embedding features at the patch level. We downloaded the model weights of [UNI](https://huggingface.co/MahmoodLab/UNI) from Hugging Face, which saved in **./UNI_weights/pytorch_model.bin**. 

```bash
python HE_sub_patches_UNI_features_extraction.py
```
Extracted UNI features are saved in **../../preprocessed_data/HE_images/sub_HE_patches_UNI**. 

##### 2. mIHC data preprocessing
In **./preprocessing/mIHC_data** directory

###### 2.1 mIHC quantitative features extraction

For mIHC TMAs slide after digitization, we used a pathology analysis software HALO (Indica Labs) to conduct cell segmentation and features extraction. The features extracted from the mIHC data of each core include the cell position, optical features of stained cells, and morphological information. The optical features represent a quantitative depiction of the markers CD68, CD163, CD8, FOXP3, PD-L1, and panCK. For proteins expressed in the cytoplasm or on the cell surface, such as CD68, CD163, CD8, PD-L1, and panCK, the optical features include Positive Classification (binary), Positive Cytoplasm Classification (binary), Cell Intensity (numerical), and Cytoplasm Intensity (numerical). For the protein FOXP3, which is expressed in the cell nucleus, the optical features include Positive Classification, Positive Nucleus Classification, Cell Intensity, and Nucleus Intensity. For numerical features including Cytoplasm Intensity, Nucleus Intensity, and Cell Intensity, **Min-Max Scaling** is applied to each TMA to bring them within the range of 0-1. The morphological information includes Cytoplasm Area, Nucleus Area, Nucleus Perimeter, and Nucleus Roundness. For each cell, in addition to the x, y coordinates, there are a total of 28 features. The cell features extracted from mIHC data corresponding to each sample core are stored as separate csv files. The specific format is shown in the table below.

<div style="overflow-x: auto;">
  <table>
    <thead>
      <tr>
        <th>X_coor</th>
        <th>Y_coor</th>
        <th>PDL1 Positive Classification</th>
        <th>PDL1 Positive Cytoplasm Classification</th>
        <th>PDL1 Cytoplasm Intensity</th>
        <th>PDL1 Cell Intensity</th>
        <th>CD68 Positive Classification</th>
        <th>CD68 Positive Cytoplasm Classification</th>
        <th>CD68 Cytoplasm Intensity</th>
        <th>CD68 Cell Intensity</th>
        <th>FOXP3 Positive Classification</th>
        <th>FOXP3 Positive Nucleus Classification</th>
        <th>FOXP3 Nucleus Intensity</th>
        <th>FOXP3 Cell Intensity</th>
        <th>CD8a Positive Classification</th>
        <th>CD8a Positive Cytoplasm Classification</th>
        <th>CD8a Cytoplasm Intensity</th>
        <th>CD8a Cell Intensity</th>
        <th>CD163 Positive Classification</th>
        <th>CD163 Positive Cytoplasm Classification</th>
        <th>CD163 Cytoplasm Intensity</th>
        <th>CD163 Cell Intensity</th>
        <th>PANCK Positive Classification</th>
        <th>PANCK Positive Cytoplasm Classification</th>
        <th>PANCK Cytoplasm Intensity</th>
        <th>PANCK Cell Intensity</th>
        <th>Cytoplasm Area</th>
        <th>Nucleus Area</th>
        <th>Nucleus Perimeter</th>
        <th>Nucleus Roundness</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2391</td>
        <td>2492</td>
        <td>0</td>
        <td>0</td>
        <td>0.036498</td>
        <td>0.016839</td>
        <td>0</td>
        <td>0</td>
        <td>0.029290</td>
        <td>0.010089</td>
        <td>0</td>
        <td>0</td>
        <td>0.018969</td>
        <td>0.020018</td>
        <td>1</td>
        <td>1</td>
        <td>0.230719</td>
        <td>0.213315</td>
        <td>0</td>
        <td>0</td>
        <td>0.026353</td>
        <td>0.009497</td>
        <td>0</td>
        <td>0</td>
        <td>0.032542</td>
        <td>0.027646</td>
        <td>0.219150</td>
        <td>0.317768</td>
        <td>0.219824</td>
        <td>0.748344</td>       
      </tr>   
      <tr>
        <td>5003</td>
        <td>1466</td>
        <td>0</td>
        <td>0</td>
        <td>0.092665</td>
        <td>0.069642</td>
        <td>1</td>
        <td>1</td>
        <td>0.227585</td>
        <td>0.172921</td>
        <td>0</td>
        <td>0</td>
        <td>0.022205</td>
        <td>0.023716</td>
        <td>0</td>
        <td>0</td>
        <td>0.126692</td>
        <td>0.049233</td>
        <td>0</td>
        <td>0</td>
        <td>0.044167</td>
        <td>0.023657</td>
        <td>1</td>
        <td>1</td>
        <td>0.088081</td>
        <td>0.084531</td>
        <td>0.109575</td>
        <td>0.284895</td>
        <td>0.209356</td>
        <td>0.766435</td>       
      </tr>
  </table>
</div>

###### 2.2 mIHC Cell-Hypergraph construction

We split each mIHC sample core area into 4×4 non-overlapping patches, and excluded patches with fewer than 100 cells.  For each patch area, we built a Cell-Hypergraph G=(V,E,W) separately, where V represents a node set, E represents a hyperedge set, and W represents a diagonal matrix of edges. A hyperedge e is constructed for each cell as the core, connecting this cell and all other cells within a 20μm radius. 

```bash
python mIHC_graph_split_construction.py
```
Constructed Cell-Hypergraphs are saved in **../../preprocessed_data/mIHC_data/constructed_mIHC_graph_split**. 

##### 3 Clinical data preprocessing
###### 3.1 Raw clinical data format

The original clinical data format is as shown in the table below, including information such as TMA_ID, sample_ID, surv_time, etc. This file is saved as **../../preprocessed_data/clinical_data/raw_clinical_data.csv**.

<div style="overflow-x: auto;">
  <table>
    <thead>
      <tr>
        <th>TMA_ID</th>
        <th>sample_ID</th>
        <th>surv_time</th>
        <th>status</th>
        <th>gender</th>
        <th>age</th>
        <th>grade</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>TMA1</td>
        <td>A-1</td>
        <td>36.0</td>
        <td>death</td>
        <td>female</td>
        <td>67.0</td>
        <td>High</td>    
      </tr>   
      <tr>
        <td>TMA5</td>
        <td>J-13</td>
        <td>72.0</td>
        <td>censored</td>
        <td>male</td>
        <td>79.0</td>
        <td>Low</td>   
      </tr>
  </table>
</div>

###### 3.2 Survival label discretization

We discretized the survival time into intervals based on the 25%, 50%, and 75% quantiles of uncensored patients in the training set. The file with discrete survival labels is saved as **../../preprocessed_data/clinical_data/surv_bin_clinical_data.csv**.

```bash
python survival_bin_preprocessing.py
```

###### 3.3 Patch-level survival label generation
Each patch's survival label inherits the discretized survival label from the patient level. 

- For HE patches, the file with patch-level discrete survival labels is saved as  **../../preprocessed_data/clinical_data/surv_bin_HE_Virchow_patches_df.csv**.
```bash
python HE_patch_label_generation.py
```

- For mIHC data, the file with patch-level discrete survival labels is saved as  **../../preprocessed_data/clinical_data/surv_bin_mIHC_patches_df.csv**.
```bash
python mIHC_patch_label_generation.py
```

### II. Model training and validation

##### 1. Multi-step multi-modal ensemble survival model (MMES) training and validation
###### 1.1 HE modality step1 & step2

Step1: Patch-level single-modality discrete survival prediction 

Step2: Patient-level discrete-time hazard rate ensemble 

###### 3.2 mIHC modality step1 & step2

Step1: Patch-level single-modality discrete survival prediction 

Step2: Patient-level discrete-time hazard rate ensemble 

###### 3.3 Step3: Cox regression based on HE and mIHC ensembled scores and clinical data 

### Reference

[1] Vorontsov, E.; Bozkurt, A.; Casson, A.; Shaikovski, G.; Zelechowski, M.; Severson, K.; Zimmermann, E.; Hall, J.; Tenenholtz, N.; Fusi, N. A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine 2024, 1-12.

[2] Chen, R. J.; Ding, T.; Lu, M. Y.; Williamson, D. F.; Jaume, G.; Song, A. H.; Chen, B.; Zhang, A.; Shao, D.; Shaban, M. Towards a general-purpose foundation model for computational pathology. Nature Medicine 2024, 30 (3), 850-862.