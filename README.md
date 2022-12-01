# Fake News Detection
<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="Python" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />  <img alt="Numpy" 
src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" />  <img alt="Pandas" 
src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img alt="Scikit-Learn" 
src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=00b35a&labelColor=00b35a" /> <img alt="Pycharm" 
src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> 

## Description
As people spending more time online, misinformation has become a real issue. People that are less rational could be easily mislead by the false information and it would cause some bad consiquences sometimes. let alone the fake news headlines as a click bait publish by the mainstream medias in order to get more views.\
To solve this problem, people have came up with many solutions. One of which is using machine learning to identify and eliminate the fake news headlines.\
For the purpose of this project, you will find an efficient method to solve this problem which is using the decision tree algorithm
 

### Dataset
There are two sets of data. One includes the fake news headlines and the other includes real new headlines
- 3266 examples of new headlines, 5799 features
- 1298 fake news headlines and 1968 real news headlines
- 70% training set, 15% validation set, 15% test set

## Decision Tree Classifier
### Gini Criteria  
- Best Depth: 20
- Train_acc: 0.9444, Val_acc: 0.8163, Test_acc: 0.8041
### Entropy Criteria  
- Best Depth: 10
- Train_acc: 0.8771, Val_acc: 0.8102, Test_acc: 0.8143
