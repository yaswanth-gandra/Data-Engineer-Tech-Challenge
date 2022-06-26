#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[5]:


import warnings

warnings.filterwarnings('ignore')


# In[6]:


data = '/Users/priyash/26-06-2022/car_evaluation.csv'

df = pd.read_csv(data, header=None)


# In[7]:


df.shape


# In[8]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df.columns = col_names

col_names


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:
    
    print(df[col].value_counts())   


# In[14]:


df['class'].value_counts()


# In[15]:


df.isnull().sum()


# In[16]:


X = df.drop(['class'], axis=1)

y = df['class']


# In[17]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[18]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[19]:


# check data types in X_train

X_train.dtypes


# In[20]:


X_train.head()


# In[21]:


# import category encoders

import category_encoders as ce


# In[22]:


# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[23]:


X_train.head()


# In[24]:


X_test.head()


# In[25]:


# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier


# In[26]:


# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)


# In[27]:


y_pred_gini = clf_gini.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


# In[29]:


y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini


# In[30]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


# In[31]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


# In[32]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train)) 


# In[33]:


pip install graphviz


# In[35]:


import graphviz 
dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 


# In[36]:


# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)


# In[37]:


y_pred_en = clf_en.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))


# In[39]:


y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en


# In[40]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# In[41]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))


# In[42]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_en.fit(X_train, y_train)) 


# In[43]:


import graphviz 
dot_data = tree.export_graphviz(clf_en, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 


# In[44]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)


# In[46]:


# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))

