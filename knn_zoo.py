#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
zoo=pd.read_csv("Zoo.csv")
zoo.head()


# In[2]:


zoo.tail()


# In[3]:


zoo.shape


# In[4]:


zoo.isnull().any(axis=1)


# In[5]:


zoo.isnull().sum()


# In[6]:


zoo["legs"].unique()


# In[7]:


zoo["type"].unique()


# # Test size=0.2

# In[8]:


# Training and Test data split from zoo
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2) 


# In[9]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier as KNN
model=KNN(n_neighbors=3)


# In[12]:


model.fit(train.iloc[:,1:16],train.iloc[:,17])


# In[13]:


import numpy as np
train_acc = np.mean(model.predict(train.iloc[:,1:16])==train.iloc[:,17])
print(train_acc)


# In[14]:


test_acc = np.mean(model.predict(test.iloc[:,1:16])==test.iloc[:,17]) 
print(test_acc)


# In[15]:


#For 5 nearest neighbors
model1=KNN(n_neighbors=5)


# In[16]:


model1.fit(train.iloc[:,1:16],train.iloc[:,17])


# In[17]:


train_acc1=np.mean(model1.predict(train.iloc[:,1:16])==train.iloc[:,17])
train_acc1


# In[18]:


test_acc1 = np.mean(model1.predict(test.iloc[:,1:16])==test.iloc[:,17]) 
test_acc1


# In[19]:


acc=[]
 
for i in range(3,50,2):
    neigh = KNN(n_neighbors=i)
    neigh.fit(train.iloc[:,1:16],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:16])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:16])==test.iloc[:,17])
    acc.append([train_acc,test_acc])


# In[20]:



import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])


# In[24]:


#For n=3
model1.predict(test.iloc[:,1:16])


# In[25]:


#for n=5
model.predict(test.iloc[:,1:16])


# In[ ]:





# # Test size=0.3

# In[26]:


# Training and Test data split from zoo
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.3) 


# In[27]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier as KNN
model=KNN(n_neighbors=3)


# In[28]:


model.fit(train.iloc[:,1:16],train.iloc[:,17])


# In[29]:


train_acc = np.mean(model.predict(train.iloc[:,1:16])==train.iloc[:,17])
print(train_acc)


# In[30]:


test_acc1 = np.mean(model1.predict(test.iloc[:,1:16])==test.iloc[:,17]) 
test_acc1


# In[31]:


#predicted
model.predict(test.iloc[:,1:16])


# In[32]:


#actual
test.iloc[:,17]


# In[ ]:




