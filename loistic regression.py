#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd      
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


deabets_data = pd.read_csv(r'C:\Users\Eng.Amira\Desktop\data\deabets.csv')


# In[6]:


deabets_data.head()


# In[7]:


deabets_data.tail()


# In[8]:


deabets_data.shape


# In[9]:


deabets_data.info


# In[10]:


deabets_data.isnull().sum()


# In[11]:


deabets_data.describe()


# In[17]:


deabets_data['Diabetes_binary'].value_counts()


# In[18]:


X = deabets_data.drop(columns='Diabetes_binary', axis=1)
Y = deabets_data['Diabetes_binary']


# In[19]:


print(X)


# In[20]:


print(Y)


# In[37]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[29]:


model = LogisticRegression()


# In[38]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2,5,2,140,62,0,3,160,23)


# In[36]:


# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Diabetes Disease')
else:
  print('The Person has Diabetes Disease')
  


# In[ ]:





# In[ ]:




