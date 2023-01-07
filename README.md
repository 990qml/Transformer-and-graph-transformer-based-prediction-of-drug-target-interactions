# Transformer-and-graph-transformer-based-prediction-of-drug-target-interactions

## Introduction
   Identifying Drug-Target Interaction (DTI) is of great benefit to drug discovery and drug reuse. As we all know, the discovery of new drugs takes a long time and costs a lot, which has forced people to consider using better methods to find drugs. In the past period, researchers have made great progress in applying Deep Learning (DL) to the development of DTI. How to further improve the ability of deep learning in this aspect has become a research hotspot. Therefore, we propose a model based on deep learning, which applies Transformer to DTI prediction. The model uses Transformer and Graph Transformer to extract the feature information of protein and compound molecules respectively, and combines their respective representations to predict interactions. We use Human and C.elegans, the two benchmark datasets, evaluated the proposed method in different experimental settings and compared it with the latest DL model. The results show that the proposed model based on DL is an effective method for classification and recognition of DTI prediction, and its performance on the two data sets is significantly better than other DL based methods.
 ## Installation
1、Environment configuration requirements: Install (Pytorch 0.4.0) (https://pytorch.org/). Also need to install yaml.The code has been tested with Python 3.8.

2、Datasets: The data processing and configuration are described in the datasets folder. The datasets are the Human dataset and the C.elegans dataset

3、Run: Configuration file: First, you need to modify the variables DATASET and type in the data_process.py file to pre-process the data in the dataset, and then train the model in the train.py file.
