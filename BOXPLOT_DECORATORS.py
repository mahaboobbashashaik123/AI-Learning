#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('Boston.csv')


# In[3]:


data


# In[4]:


sns.boxplot(x=data["tax"])


# In[5]:


y=sns.boxplot(x=data["ptratio"])


# In[6]:


sns.boxplot(x=data["rad"])


# In[7]:


sns.boxplot(x=data["age"])


# In[8]:


#boxplot with decorators


# In[9]:


def boxplot1(func):
    def boxplot2():
        print(sns.boxplot(x=data["ptratio"]))
        func()
        print("Boxplot below")
    return boxplot2()


# In[10]:


@boxplot1
def boxplot3():
    print("Boxplot with decorators")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




