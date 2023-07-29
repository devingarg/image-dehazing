# image-dehazing
Exploring Image Dehazing


## Set up

All the code is available in interactive jupyter notebooks and can be run with minimal setup. The only requirements are: `torch`, `cv2`, `matplotlib`.


## Datasets
We work with the following datasets:

- O-Haze: a dehazing benchmark with real hazy and haze-free outdoor images. Details [here](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/).
- D-HAZY: a dataset to evaluate quantitatively dehazing algorithms. Details [here](https://ancuti.meo.etc.upt.ro/D_Hazzy_ICIP2016/).

## Techniques 
We implement the following three techniques to perform image dehazing on these datasets:

### 1. Dehazing using dark channel prior [^1]

This is a classical technique put forth by He _et al._ in [this](https://doi.org/10.1109/CVPR.2009.5206515) paper.

The implementation is present in the [dark_channel_prior/dark_channel_prior_approach.ipynb](dark_channel_prior/dark_channel_prior_approach.ipynb) notebook.

### 2. Direct dehazing using an autoencoder

We employ a deep learning-based model (autoencoder architecture) to directly generate the dehazed version of the input image. 

The implementation is present in the [direct_dehazing/autoencoder_dehazing.ipynb](direct_dehazing/autoencoder_dehazing.ipynb) notebook.

### 3. Dehazing by first predicting the transmission map

In this technique, we use the same architecture as the previous approach. However, instead of attempting to generate the dehazed image directly (which is quite clearly a tougher problem), we ask the network to only predict the depth map for the image. 

The inverse of this depth map is then used as an estimate for the transmission map. We then dehaze images using this transmission map by following the atmospheric scattering model followed by authors of the Dark Channel Prior paper mentioned above. 

Note: This technique requires ground truth depth map data which is only available in the NYU subset of the DHazy dataset. 

The implemetation is in sequentially named jupyter notebooks in the `dehazing_via_tmap` directory:

1. [dehazing_via_tmap/01_autoencoder_tmap_predictor.ipynb](dehazing_via_tmap/01_autoencoder_tmap_predictor.ipynb): Contains the train, test code for the third techniqye i.e., predicting the depth map from the hazy image.
2. [dehazing_via_tmap/02_predict_dmap.ipynb](dehazing_via_tmap/02_predict_dmap.ipynb): Contains code to predict depth maps for any set of images using the model trained in the notebook above.
3. [dehazing_via_tmap/03_dmap_to_refined_tmap.ipynb](dehazing_via_tmap/03_dmap_to_refined_tmap.ipynb): Contains the code to convert predited depth maps to transmission maps. These transmission maps are also refined using the hazy images.
4. [dehazing_via_tmap/04_dehazed_from_tmap_hazy.ipynb](dehazing_via_tmap/04_dehazed_from_tmap_hazy.ipynb): Contains the code to obtain the dehazed images by taking transmission maps and hazy images as input.



## Utility code

Following is a description of the utility files in the repository:

1. [download_datasets.ipynb](download_datasets.ipynb): Used to download the O-Haze and D-HAZY datasets for the dehazing problem.
2. [utils/model.py](utils/model.py): Defines the autoencoder architecture we use in the project.
3. [utils/loss.py](utils/loss.py): Defines the loss function used.
    We use a perceptual loss function along with mean-squared error as we are trying to generate the dehazed image in the second technique. So, that function has been defined in this file.
4. [utils/data.py](utils/data.py): Dataset classes, transforms and other data-related utilities have been defined here.





[^1]: Kaiming He, Jian Sun and Xiaoou Tang, "Single image haze removal using dark channel prior," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, 2009, pp. 1956-1963, doi: 10.1109/CVPR.2009.5206515.
