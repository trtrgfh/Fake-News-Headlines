<img src="https://github.com/trtrgfh/Fake-News-Headlines/assets/73056232/1debb1d9-acff-4a78-b57e-d6151b92ad58" width="600"/>

# Fake News Detection

# Project Overview
As online engagement increases, misinformation has become a significant problem, with less rational individuals being easily misled by false information, leading to harmful consequences. Moreover, mainstream media often publishes sensationalized fake news headlines as clickbait to attract more views. To address this issue, this project aims to employ the decision tree algorithm to identify and eliminate fake news headlines. The project will implement the methods needed for the decision tree algorithm, explore both Gini impurity and entropy criteria to optimize classification, and utilize the XGBoost library to enhance performance. By combining these methods, the project seeks to create an efficient solution for mitigating the spread of fake news, contributing to the broader effort to combat online misinformation.

# Installation and Setup
## Python Packages Used
- **Data Manipulation:** numpy, pandas 
- **Data Visualization:** matplotlib, graphviz
- **Machine Learning:** scikit-learn, xgboost

# Data
clean_fake.txt and clean_real.txt are used where clean_real.txt contains real news headlines and clean_fake.txt contains fake news headlines. 

# Results and evaluation
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200620180439/Gini-Impurity-vs-Entropy.png" width="450"/>

One downside of using only one decision tree is that small changes in the training set could result in a completely different decision tree. This could lead to a very unstable and unreliable model, and one way to address this issue is by creating multiple trees (tree ensembles) to make the algorithm more robust.

### Random Forest Algorithm Intuition
- Using sampling with replacement to create a new training set of size m
- Train a decision tree on the new dataset, and when choosing a feature to use to split, pick a random subset of k < n features for the algorithm to choose from
- Repeat the process B times (common choice: 64, 128)

### Boosted Trees Intuition
- When creating a new training set, make it more likely to pick the misclassified examples from previously trained trees
  
## Result
### Gini Criterion  
- Best Depth: 20
- Train_acc: 0.9444, Val_acc: 0.8163, Test_acc: 0.8041
### Entropy Criterion  
- Best Depth: 10
- Train_acc: 0.8771, Val_acc: 0.8102, Test_acc: 0.8143
### XGBoost
- Train_acc: 0.9641, Val_acc: 0.8449, Test_acc: 0.8429
