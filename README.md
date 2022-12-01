# Fake News Detection
<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="Python" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />  <img alt="Numpy" 
src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />  <img alt="Pandas" 
src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img alt="Scikit-Learn" 
src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=00b35a&labelColor=00b35a" /> <img alt="Pycharm" 
src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> 

## Description
As people spending more time online, misinformation has become a real issue. People that are less rational could be easily mislead by the false information and it would cause some bad consiquences sometimes. let alone the fake news headlines as a click bait publish by the mainstream medias in order to get more views. To solve this problem, people have came up with many solutions. One of which is using machine learning to identify and eliminate the fake news headlines.\
For the purpose of this project, you will find an efficient method to solve this problem which is using the decision tree algorithm\
This project will implements the methods needed for the decision tree algorithm and try out both the gini criterion and the entropy criterion, as well as the xgboost library. 

![Gini-Entropy-Differences](https://media.geeksforgeeks.org/wp-content/uploads/20200620180439/Gini-Impurity-vs-Entropy.png)

One downside of using only one decision tree is that small changes in the training set could result in a completely different decision tree, so creating multiple trees (tree ensembles) could make the algorithm more robust.

### Random Forest Algorithm Intuition
- Using sampling with replacement to create a new training set of size m
- Train a decision tree on the new dataset, and when choosing a feature to use to split, pick a random subset of k < n features for the algorithm to choose from
- Repeat the process B times (common choice: 64, 128)

### Boosted Trees Intuition
- When creating a new training set, make it more likely to pick misclassified examples from previously trained trees
  
## Result
### Gini Criterion  
- Best Depth: 20
- Train_acc: 0.9444, Val_acc: 0.8163, Test_acc: 0.8041
### Entropy Criterion  
- Best Depth: 10
- Train_acc: 0.8771, Val_acc: 0.8102, Test_acc: 0.8143
### XGBoost
- Train_acc: 0.9641, Val_acc: 0.8449, Test_acc: 0.8429
