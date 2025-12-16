# CIFAR-10-Classification
Classification of the CIFAR 10 dataset using CNN
CIFAR-10 Image Classification with Convolutional Neural Networks
 Project Overview

This project implements a custom Convolutional Neural Network (CNN) from scratch for image classification on the CIFAR-10 dataset. The goal is to design a model that generalizes well to unseen data while avoiding overfitting, using modern deep learning practices rather than pretrained architectures.

**Objectives**

--Build and train a CNN for 32×32 RGB image classification
--Reduce overfitting through architectural and regularization techniques
--Evaluate performance using accuracy, confusion matrix, and class-wise metrics
--Analyze common misclassification patterns

**Model Architecture**

The CNN consists of:
--Stacked 3×3 convolutional layers with ReLU activation
--Max pooling layers for spatial downsampling
--Global Average Pooling to reduce parameter count and improve generalization
--Fully connected layers with Dropout for regularization
--This design avoids large convolution kernels and excessive dense layers, which are known to perform poorly on low-resolution images.

**Training Strategy**

--To improve generalization, the following techniques were applied:
--Data augmentation (random flips, shifts, rotations, zoom)
--EarlyStopping to prevent overfitting
--ReduceLROnPlateau to adapt the learning rate
--Adam optimizer with He initialization

**Results**
--Training Accuracy: ~84%    
--Validation Accuracy: ~82%    
--Test Accuracy: ~82%

**Evaluation & Analysis**
Model performance was evaluated using:

--Test set accuracy
--Confusion matrix
--Classification report
--Visualization of misclassified samples

**Dataset**
--CIFAR-10
--60,000 images across 10 classes
--32×32 RGB resolution

**Technologies Used**
--Python
--TensorFlow / Keras
--NumPy, Matplotlib
--scikit-learn
