#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


pwd


# In[3]:


data=pd.read_csv("E:\\Machine_Learning_AZ_Template_Folder\\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data_Preprocessing\\Data_Preprocessing\\Data.csv")


# In[4]:


data


# In[5]:


X=data.iloc[:,:-1]#its an independent variables where -1 is and exclude last column.
y=data.iloc[:,3]


# # 1. TAKING CARE OF MISSING VALUES!

# In[6]:


from sklearn.preprocessing import Imputer


# In[7]:


imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)


# In[8]:


imputer=imputer.fit(X.iloc[:,1:3])


# In[9]:


X.iloc[:,1:3]=imputer.transform(X.iloc[:,1:3])


# # 2. Encoding The Categorial Variables!

# In[29]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[11]:


labelencoder_X=LabelEncoder()


# In[12]:


X.iloc[:,0]=labelencoder_X.fit_transform(X.iloc[:,0])


# In[30]:


onehotencoder=OneHotEncoder(categorical_features=[0])


# In[31]:


labelencoder_y=LabelEncoder()


# In[32]:


y=labelencoder_y.fit_transform(y)


# # 3. SPLIT THE DATA SET INTO TRAIN AND TEST.

# In[33]:


from sklearn.model_selection import train_test_split 


# In[34]:


X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2, random_state=0)


# # 4.FEATURE SCALING

# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


sc_X=StandardScaler()


# In[37]:


X_train=sc_X.fit_transform(X_train)


# In[38]:


X_test=sc_X.transform(X_test)


# In[ ]:




