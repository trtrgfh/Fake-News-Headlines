#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# load the data
df = pd.read_csv("/Users/yehao/Desktop/Projects/Machine Learning Practice/fake news detector/news.csv")
df.shape


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


# get the labels of the data
labels = df.label
labels.head()


# In[6]:


# split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 42)


# In[7]:


X_train.shape


# In[8]:


# initialize a TfidfVectorizer
# stop_words are the words you want to filter out in the data.
# max_df sets the maximum of frequency, any common words appear more than 70% of the time would be filter out. 
vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# fit will calculate the mean and standard deviation, then transform will make the data standardized (x-µ/σ)
# then we will need to use the same mean and sd of the training set on test set.
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
# We use the test dataset to get a good estimate of how our model performs on any new data.
# The new unseen data could be just 1 data point
# That’s why we need to keep and use the training data parameters for scaling the test set.


# In[9]:


# each row is a text, and each column the value whether a unique word in all the text exist in this row of text.
X_train.shape


# In[10]:


vectorizer.get_feature_names_out().shape


# In[11]:


# initialize a PassiveAggressiveClassifier
# PAC is useful when there is a huge amount of data and it is computationally infeasible to train the entire dataset.
# max_iter is the maximum number of passes over the training data (aka epochs)
# Assume 200 samples, batch size is 5. Then you will have 40 batches. 
# Weights will be updated after each batch of 5 samples. 
# One epoch will involve 40 batches or 40 updates to the model.
# With 50 epochs, the model will pass through the whole dataset 50 times.
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train,y_train)


# In[18]:


# use the pac model to predict the outcomes of X_test
y_pred=pac.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[22]:


# get the confusion_matrix between y_test(the actual output of the test set) and y_pred(the predicted output)
confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
# TP = 589, TN = 587, FP = 42, and FN = 49.
# Accuracy Score = (TP+TN)/ (TP+FN+TN+FP)
# TP is the predicted is fake news and actual is also fake news.


# In[ ]:




