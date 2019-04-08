# MsDBP
Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network

we have reported a novel predictor MsDBP, a DNA-binding protein prediction method that combines multi-scale sequence features into a deep neural network. For a given protein, we first divide the entire protein sequence into 4 subsequences to extract multi-scale features, and then we regard the feature vector as the input of the network and apply a branch of dense layers to automatically learn diverse hierarchical features. Finally, we use a neural network with two hidden layers to connect their outputs for DBPs prediction.

Dependency:
Python 3.6.2
Numpy 1.13.1
Scikit-learn 0.19.0
Tensorflow 1.3.0
keras 2.0.8

Usage:
Run this file from command line.
For example:
python MsDBP_predict.py

Reference:
Xiuquan Du, Yanyu Diao, Heng Liu, Shuo Li. MsDBP: Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network. Submitted.
