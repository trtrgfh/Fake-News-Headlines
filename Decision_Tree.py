import numpy as np
import matplotlib.pyplot as plt
from public_tests import *


# Calculate entropy
# $$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$

def compute_entropy(y):
    """
    Return the entropy 
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

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into left and right branches    
    """
    
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices


# 
# Information Gain = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$
# 
# where 
# - $H(p_1^\text{node})$ is entropy at the node 
# - $H(p_1^\text{left})$ and $H(p_1^\text{right})$ are the entropies at the left and the right branches resulting from the split
# - $w^{\text{left}}$ and $w^{\text{right}}$ are the proportion of examples at the left and right branch respectively

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
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


def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value to split the node data 
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

