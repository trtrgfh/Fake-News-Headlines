#!/usr/bin/env python
# coding: utf-8

# # Decision Trees
# 
# Implement a decision tree from scratch and apply it to the task of classifying whether a mushroom is edible or poisonous.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from public_tests import *


# <a name="2"></a>
# ## 2 -  Problem Statement
# 
# Suppose you are starting a company that grows and sells wild mushrooms. 
# - Since not all mushrooms are edible, you'd like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
# - You have some existing data that you can use for this task. 
# 
# Can you use the data to help you identify which mushrooms can be sold safely? 
# 
# Note: The dataset used is for illustrative purposes only. It is not meant to be a guide on identifying edible mushrooms.
# 
# 
# 
# <a name="3"></a>
# ## 3 - Dataset
# 
# You will start by loading the dataset for this task. The dataset you have collected is as follows:
# 
# | Cap Color | Stalk Shape | Solitary | Edible |
# |:---------:|:-----------:|:--------:|:------:|
# |   Brown   |   Tapering  |    Yes   |    1   |
# |   Brown   |  Enlarging  |    Yes   |    1   |
# |   Brown   |  Enlarging  |    No    |    0   |
# |   Brown   |  Enlarging  |    No    |    0   |
# |   Brown   |   Tapering  |    Yes   |    1   |
# |    Red    |   Tapering  |    Yes   |    0   |
# |    Red    |  Enlarging  |    No    |    0   |
# |   Brown   |  Enlarging  |    Yes   |    1   |
# |    Red    |   Tapering  |    No    |    1   |
# |   Brown   |  Enlarging  |    No    |    0   |
# 
# 
# -  You have 10 examples of mushrooms. For each example, you have
#     - Three features
#         - Cap Color (`Brown` or `Red`),
#         - Stalk Shape (`Tapering` or `Enlarging`), and
#         - Solitary (`Yes` or `No`)
#     - Label
#         - Edible (`1` indicating yes or `0` indicating poisonous)
# 
# <a name="3.1"></a>
# ### 3.1 One hot encoded dataset
# For ease of implementation, we have one-hot encoded the features (turned them into 0 or 1 valued features)
# 
# | Brown Cap | Tapering Stalk Shape | Solitary | Edible |
# |:---------:|:--------------------:|:--------:|:------:|
# |     1     |           1          |     1    |    1   |
# |     1     |           0          |     1    |    1   |
# |     1     |           0          |     0    |    0   |
# |     1     |           0          |     0    |    0   |
# |     1     |           1          |     1    |    1   |
# |     0     |           1          |     1    |    0   |
# |     0     |           0          |     0    |    0   |
# |     1     |           0          |     1    |    1   |
# |     0     |           1          |     0    |    1   |
# |     1     |           0          |     0    |    0   |
# 
# Therefore,
# - `X_train` contains three features for each example 
#     - Brown Color (A value of `1` indicates "Brown" cap color and `0` indicates "Red" cap color)
#     - Tapering Shape (A value of `1` indicates "Tapering Stalk Shape" and `0` indicates "Enlarging" stalk shape)
#     - Solitary  (A value of `1` indicates "Yes" and `0` indicates "No")
# 
# - `y_train` is whether the mushroom is edible 
#     - `y = 1` indicates edible
#     - `y = 0` indicates poisonous

# In[2]:


X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])


# In[3]:


print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))


# In[4]:


print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))


# In[5]:


print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))


# ### Calculate entropy
# 
# $$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$

# In[6]:


# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    entropy = 0.
    
    p1 = 0
    if len(y) != 0:
        p1 = np.sum(y == 1) / len(y)

    if p1 != 0 and p1 != 1:
        entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    else:
        entropy = 0. 
    
    return entropy


# In[7]:


# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (ndarray):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (ndarray): Indices with feature value == 1
        right_indices (ndarray): Indices with feature value == 0
    """
    
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices


# Ccompute the `compute_information_gain()` 
# 
# $$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$
# 
# where 
# - $H(p_1^\text{node})$ is entropy at the node 
# - $H(p_1^\text{left})$ and $H(p_1^\text{right})$ are the entropies at the left and the right branches resulting from the split
# - $w^{\text{left}}$ and $w^{\text{right}}$ are the proportion of examples at the left and right branch respectively

# In[8]:


# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)

    weighted_entropy = w_left * left_entropy + w_right * right_entropy

    information_gain = node_entropy - weighted_entropy
    
    return information_gain


# In[10]:


root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)
    
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

compute_information_gain_test(compute_information_gain)


# In[11]:


# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    num_features = X.shape[1]
    
    best_feature = -1
    
    max_info_gain = 0
    
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
   
    return best_feature


# In[ ]:




