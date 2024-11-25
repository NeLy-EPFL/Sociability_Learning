# Sociability Learning
Code to analyze the data from the paper 
"Conspecific sociability is regulated by associative learning circuits"
Victor Lobato-Rios, Thomas Ka Chung Lam, Pavan Ramdya
2024

## Installation
- Create a conda environment with python 3.7 and sleap version 1.2.8:

`conda create -y -n sociability -c sleap -c nvidia -c conda-forge sleap=1.2.8 python=3.7`

- Clone this repository:
  
`$git clone https://github.com/NeLy-EPFL/Sociability_Learning.git`

- Go into the repository folder and activate de conda environment:
  
`$cd Sociability_Learning`

`$conda activate sociability`

- Install this repository as a package:
  
`$pip install –e .`

## Usage

The folder `scripts` contains examples of using the `Sociability_Learning` package. These scripts replicate the analysis and obtain the plots from the paper.

- `compare_model_with_random_networks.py`: compares the hits from our network (defined in the script `Sociability_Learning/utils_connectomics.py`) with randomly generated networks that conserve or not the proportion of neuronal `hits' for each brain region **(Fig. 3e)**.
- `embedding.ipynb`: generates the UMAP embedding based on the proximity events from the female-female control experiments **(Fig. 1e-f; EDFig. 2)**.
- `folders_to_process_*.yaml`: list of data to analyze. Data can be downloaded from our [Dataverse](https://dataverse.harvard.edu/dataverse/sociability_learning/).
- `generate_figures_videos.sh`: runs several scripts to generate Fig. 1e-f, EDFig. 2, and videos 3-7.
- `get_sociability_index.py`: computes the sociability index from the data specified in `folders_to_process_behavior.yaml`. Several parameters, as the selection of the control experiments to compute the metric's thresholds, are at the top of the script **(Fig. 1h-k; Fig. 2, Fig. 3b-c, EDFig. 5, 7, 8)**.
- `get_sociability_model.py`: obtains the network formed in the connectome by the cell types specified in the script `Sociability_Learning/utils_connectomics.py`.
- `preprocess_data_from_2p_setup.py`: obtains the Delta F/F from the neural activity recordings, the treadmill ratations from Fictrac, and the location of the free-moving fly from SLEAP.
- `twop_analysis.py`: analyzes the neural and behavioral data from the two-photon recordings specified in `folders_to_process_.yaml`.
- `video*.py`: compiles the videos specified in the name of the script.
