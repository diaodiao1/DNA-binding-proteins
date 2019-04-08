# MsDBP
Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network

we have reported a novel predictor MsDBP, a DNA-binding protein prediction method that combines multi-scale sequence features into a deep neural network. For a given protein, we first divide the entire protein sequence into 4 subsequences to extract multi-scale features, and then we regard the feature vector as the input of the network and apply a branch of dense layers to automatically learn diverse hierarchical features. Finally, we use a neural network with two hidden layers to connect their outputs for DBPs prediction.

Dependency:</br>
Python 3.6.2</br>
Numpy 1.13.1</br>
Scikit-learn 0.19.0</br>
Tensorflow 1.3.0</br>
keras 2.0.8</br>

Usage:</br>
Run this file from command line.</br>
For example:</br>
python MsDBP_predict.py</br>

Contact us:</br>
dxqllp@163.com

Reference:</br>
Xiuquan Du, Yanyu Diao, Heng Liu, Shuo Li. MsDBP: Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network. Submitted.
