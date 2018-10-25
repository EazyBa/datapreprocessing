#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


pwd


# # IMPORTING THE DATA

# In[3]:


data=pd.read_csv("E:\\Machine_Learning_AZ_Template_Folder\\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data_Preprocessing\\Data_Preprocessing\\Data.csv")


# In[4]:


data


# In[5]:


X=data.iloc[:,:-1].values#its an independent variables where -1 is and exclude last column.
y=data.iloc[:,3].values


# # 3. SPLIT THE DATA SET INTO TRAIN AND TEST.

# In[19]:


from sklearn.model_selection import train_test_split 


# In[21]:


X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2, random_state=0)


# # 4.FEATURE SCALING

# In[22]:


"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""


# In[ ]:




