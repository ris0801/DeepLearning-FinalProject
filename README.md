# DeepLearning Final Project

## Table of Contents
- [Aim](#aim)
- [Experiments Performed](#experiments-performed)
- [Models Utilized](#models-utilized)
- [Results](#results)
- [Team Members](#team-members)
- [Instructions to Run the Python Files](#instructions-to-run-the-python-files)

## Aim
The objective of this project is to explore and understand the key differentiators between neural networks that demonstrate strong generalization capabilities and those that do not.

## Experiments Performed
1. **Impact of Explicit Regularization**: Assessing how explicit regularization techniques, such as data augmentation, weight decay, and dropout, affect the performance and generalization of deep learning models.
2. **Influence of Implicit Regularization**: Examining the role of implicit regularization methods, particularly Batch Normalization (BatchNorm), in enhancing the training and generalization outcomes of neural networks.
3. **Effects of Input Data Corruption**: Investigating how various forms of input data corruption, including pixel shuffling, addition of Gaussian noise, and insertion of random pixels, impact the training efficiency and convergence of deep learning models.
4. **Impact of Label Corruption**: Analyzing how label corruption at different levels (ranging from 1% to 100%) influences the training dynamics and generalization ability of neural networks.

## Models Utilized
- AlexNet
- Inception_v3
- Wide ResNet
- Inception (tiny)
- MLP-512

## Results
All the obtained results, including detailed analyses and visualizations, are provided in the "Results" section of the accompanying report.

## Team Members
- Rishabh Singh (rbs7261)
- Sarthak Chowdhary (sc9865)
- Samir Ahmed Talkal (st4703)

## Instructions to Run the Python Files
All the experiments were conducted on the NYU HPC cluster, and the results were visualized using Google Colab. 

Each Python file is standalone and can independently produce results. These files are designed to be executed separately in Google Colab. Each file includes the necessary code to reproduce all the graphs and analyses presented in the "Results" section of the report.
