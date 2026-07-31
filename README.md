# Teeth-Disease-Classification-Using-Transfer-Learning-with-ResNet50

## A transfer learning model using a pretrained ResNet50 (ImageNet weights), fine-tuned and deployed an an interactive
## Streamlit Web Application


## Overview

This project classifies images in 7 different classes of teeth diseases:

1. CaS
2. CoS
3. Gum
4. MC
5. OC
6. OLP
7. OT


## Architecture

A ResNet50 pretrained model with ImageNet weights

- *Backbone* : Pretrained convolutional layers, frozen initially.
- *Custom Head* : Trained a custom head for our dataset.
- *Two Phase Training* :
- 1. Train only the custom head with the backbone fully frozen.
  2. Unfreeze the last 30 layers of the backbone and fine-tune with a lower learning rate, keeping early low level
     ImageNet features intact while adapting higher level features to the dental domain.

## Data Pipeline

-  Images load through Tensorflow/Keras , resized to 224x224.
- *Data Augmentation*: Random horizontal flips, rotations, zoom, brightness, contrast applied to training set to expand it.
- *Normalization* : Used (preprocess_input) from ResNet50, matches the exact normalization that the pretrained weights expect
  and not a simple /255 rescale normalization style.
- *Performance Optimization*: .cache() + .prefetch() applied on dataset splits to avoid bottlenecks during training.
- *Class Weights* : Computed from the training set class , used to help with class imbalance.
- *Mixed Precision*: Using both 16-bit and 32-bit floating-points in a model to make training faster and use less memory

## Training

- *Optimizer* : Adam
- *Loss* : Sparse Categorical Crossentropy
- *Epochs* : up to 30 with:
  1. EarlyStopping with a patience = 5 + restores best weights
  2. ReduceLROnPlateau that divides learning rate in half when validation loss plateaus


## Test Results


| Class | Precision | Recall | F1-score |
|---|---|---|---|
| CaS | 0.96 | 0.99 | 0.98 |
| CoS | 0.97 | 0.99 | 0.98 |
| Gum | 0.95 | 0.99 | 0.97 |
| MC  | 0.98 | 0.86 | 0.92 |
| OC  | 0.85 | 0.98 | 0.91 |
| OLP | 0.95 | 0.94 | 0.95 |
| OT  | 0.99 | 0.95 | 0.97 |

**Macro avg F1:** 0.95 | **Weighted avg F1:** 0.95

**Numbers can change, this is only one singular instance of training and testing but other tries should give similar outputs**



## Project Structure

- Teeth_Disease_Classification.ipynb #Notebook with ResNet50 Transfer learning model
- teeth_resnet50_model.keras #Model used in application
- requirements.txt #Python Dependencies
- class_names.txt #Class names downloaded and used in application
- app.py #Streamlit Web Application
- .gitattributes # This project uses Git LFS since the model is above allowed upload limit
- README.md # Project description

## Notes:

This application is made for educational purposes. It is not a certified medical diagnosis tool and should not be used to real life healthcare decisions.


## Made by : NoorAldeen Faruq Al-Hara
