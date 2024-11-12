# Robust and Efficient Temporal Convolution network (RE-TCN)
This is the official repository for **Action Recognition in Real-World Ambient Assisted Living Environment**, our paper submitted to the Big Data Mining and Analytics

# Prerequisites

Use the following guide to set up the training environment. 

1. Create a conda environment
2. Install pytorch
3. Then install the following dependencies

**Dependencies**
```
python3 -m pip install scikit-learn
python3 -m pip install matplotlib
python3 -m pip install seaborn
python3 -m pip install PyYAML
python3 -m pip install tensorboardX
python3 -m pip install tqdm
python3 -m pip install h5py

```

Alternatively, you can use our environment.yml file to create an environment with all dependencies. Note that this environment was created on Ubuntu 22.04.4 LTS with cuda version 11.4.

```
Create the environment using the following command

conda env create -f environment.yml

Then, activate it using the following command

conda activate GCN

```

# Data Preparation

## Download datasets.

### There are three datasets to download:

1. **NTU RGB+D 60**: [Download the Skeleton dataset here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NW-UCLA**: [Download the dataset here](https://www.dropbox.com/scl/fi/6numm9wzu1cixw8nyzb91/all_sqe.zip?rlkey=it1ruxtsm4rggxldbbbr4w3yj&e=1&dl=0)
3. **SHREC'17**: [Download the dataset here](http://www-rech.telecom-lille.fr/shrec2017-hand/)

## Data Processing

### Directory Structure

Put the downloaded data into the following directory structure.

```
- data/
  - NW-UCLA/
    - all_sqe
      ...
  - ntu/
    - nturgbd_raw/
	  - nturgb+d_skeletons
            ...
  - shrec/
    - shrec17_dataset/
	  - HandGestureDataset_SHREC2017/
	    - gesture_1
	      ...
```

### Generating Data

**NW-UCLA dataset**

Move folder `all_sqe ` to `./data/NW-UCLA`

**NTU RGB+D 60 dataset**
```
First, extract all skeleton files to ./data/ntu/nturgbd_raw
 cd ./data/ntu
 # Get the skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the centre of the first frame
 python seq_transformation.py
```

**SHREC'17 dataset**

First, extract the downloaded dataset to `/data/shrec/shrec17_dataset`

Then, run `python gen_traindataset.py` and `python gen_testdataset.py` to prepare the dataset

# Training

### NTU RGB+D 60 dataset:

For cross-view, run `python main.py --device 0 1 --config ./config/nturgbd-cross-view/default.yaml`

For cross-subject, run `python main.py --device 0 1 --config ./config/nturgbd-cross-subject/default.yaml`

### NW-UCLA dataset:

Run `python main.py --device 0 1 --config ./config/ucla/nw-ucla.yaml`

### SHREC'17 dataset:

Run `python main.py --device 0 --config ./config/shrec17/shrec17.yaml`

# Testing

### NTU RGB+D 60 dataset:

For cross-view, run `python main.py --device 0 1 --config ./config/nturgbd-cross-view/default.yaml --phase test --weights path_to_model_weight`

For cross-subject, run `python main.py --device 0 1 --config ./config/nturgbd-cross-subject/default.yaml --phase test --weights path_to_model_weight`

### NW-UCLA dataset:

Run `python main.py --device 0 1 --config ./config/ucla/nw-ucla.yaml --phase test --weights path_to_model_weigh`

### SHREC'17 dataset:

Run `python main.py --device 0 1 --config ./config/shrec17/shrec17.yaml --phase test --weights path_to_model_weigh`
