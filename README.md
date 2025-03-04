# Graph-Generative-Ice
DSCI-498 Project: Graph generative model for radargram and internal ice layer


## Goal:

The goal of our project is to generate additional training data to make it able to evaluate model’s on different type of regions.

Currently, some radargram has 20 complete layer, some radargram has 20 layers but with NaNs, and some radargram has less than 20 layers.


## Tasks of our project:

### Task 1: Filling NaNs in current data files.

**Model Input**: 

**$X_{masked}$** : Node feature matrix with some values randomly masked (set to NaN or 0)

**Mask**: A 0-1 binary matrix, where 1 means the value is observed; 0 means the value is masked.

**Radargram image I**: The original radargram image

**Model Output**:

Y: The fully observed matrix without masking

**Loss function**:

Mean Squared Error Loss




### Task 2: Generating synthetic data files

**Model Input**:

Noise vector 

Radargram image I: The original radargram image

**Model Output**:

Synthetic feature matrix

### Task 3: Combine two tasks.

Assume we have a feature matrix (256, 15), and some value are NaNs.

First, we can fill in NaNs using the model for task 1.

Then, we can generate five additional layers using current known 15 layers, instead of generating 20 layers.




## Dataset

We will use the snow radar dataset created by CReSIS.

It contains radargram across L1, L2, L3 regions.

L1 is the test data from central Greenland assumed to be mostly dry snow zone
L2 contains echogram from the transition from Greenland Summit to the NE coast.
L3 contains echogram with poor image quality from NE Greenland coast.

