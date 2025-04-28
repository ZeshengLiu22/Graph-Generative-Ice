# Graph-Generative-Ice
DSCI-498 Project: Graph generative model for radargram and internal ice layer

## Project description:

In order to predict the deeper ice layer thickness, ih previous work, we represent each internal ice layer as a spatial graph and use a graph neural network that learns from the top 5 ice layers and predict for the underlying 15 layers. However, due to the limitations of radar measurements and various physical processes, layers may be discontinued at certain points (NaNs in the measurements) and total number of layers may vary across different locations. Previously, we do a data pre-processing that only keeps those radargrams with at least 20 layer. This is effective, but it significantly reduce the total number of valid radargrams.

In this project, we are trying to figure out whether generative model can be useful in expanding the number of valid radargrams. This includes two parts---If a data has 20 layers but with some NaNs, then we need to proper fill in the missing values. If a data has less than 20 layers and these layers may also have NaNs, we will first fill existing NaNs with some simple methods and then use generative model to generative synthetic layers.

## Dataset descriptions:

The dataset is collected using an airborne snow radar developed by CReSIS, capable of penetrating thick polar ice sheets to capture internal ice layers. Radargrams are generated from reflected signal strength, with each radargram having a depth of 1200–1700 pixels and a fixed width of 256 pixels. Internal ice layers are manually labeled, and layer thickness is calculated based on pixel differences between layer boundaries. Data is collected along various flight paths over key regions of the polar ice sheet for comprehensive spatial coverage.

## Required Packages:

**PyTorch**

**PyTorch Geometric**

hdf5

dill

tqdm

numpy

random

argparse

## How to run the code:

### Data Generation:



### Train model:

