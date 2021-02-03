#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
glass=pd.read_csv('glass.csv')
glass.head()


# In[2]:


glass.shape


# In[3]:


glass.tail()


# In[9]:


glass.isnull().any(axis=1)


# In[10]:


glass.isnull().sum()


# In[11]:


glass.isnull().any(axis=0)


# In[20]:


glass['RI'].unique().mean()


# # Test size=0.2 

# In[51]:


#Training and Testing set 
from sklearn.model_selection import train_test_split
train, test=train_test_split(glass, test_size=0.2)


# In[52]:


#for n=3
from sklearn.neighbors import KNeighborsClassifier as knn
model=knn(n_neighbors=3)


# In[53]:


model.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[54]:


import numpy as np
train_acc=np.mean(model.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[55]:


test_acc=np.mean(model.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[56]:


#predicted 
model.predict(test.iloc[:,0:8])


# In[58]:


#actual
test.iloc[:,9]


# In[59]:


#for n=5
from sklearn.neighbors import KNeighborsClassifier as knn
model1=knn(n_neighbors=5)


# In[60]:


model1.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[61]:


train_acc=np.mean(model1.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[62]:


test_acc=np.mean(model1.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[63]:


#predicted
model1.predict(test.iloc[:,0:8])


# In[65]:


#actual
test.iloc[:,9]


# In[67]:


acc=[]
for i in range(3,50,2):
    model=knn(n_neighbors=i)
    model.fit(train.iloc[:,0:8], train.iloc[:,9])
    train_acc=np.mean(model.predict(train.iloc[:,0:8])==train.iloc[:,9])
    test_acc=np.mean(model.predict(test.iloc[:,0:8])==test.iloc[:,9])
    acc.append([train_acc, test_acc])


# In[68]:


import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2), [i[0] for i in acc], "bo-")
plt.plot(np.arange(3,50,2), [i[1] for i in acc], "ro-")
plt.legend(["train", "test"])


# In[ ]:





# # Test size=0.1

# In[69]:


train1, test1=train_test_split(glass, test_size=0.1)


# In[70]:


model2=knn(n_neighbors=3)
model2.fit(train.iloc[:,0:8], train.iloc[:,9])


# In[71]:


train_acc=np.mean(model2.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[72]:


test_acc=np.mean(model2.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[73]:


#predicted
model2.predict(test.iloc[:,0:8])


# In[74]:


#actual
test.iloc[:,9]


# In[ ]:





# # To increase the accuracy

# In[77]:


#using Standard scalar
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()


# In[118]:


x=glass.drop(columns=['Type'])


# In[119]:


y=glass['Type']


# In[120]:


x=scale.fit_transform(x)


# In[121]:


x


# In[122]:


x=pd.DataFrame(x)


# In[123]:


x


# In[125]:


glass_df=pd.concat([x,y],axis=1)


# In[126]:


glass_df


# In[127]:


train,test=train_test_split(glass_df,test_size=0.2)


# In[128]:


Std_model=knn(n_neighbors=3)
Std_model.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[130]:


train_acc=np.mean(Std_model.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[131]:


test_acc=np.mean(Std_model.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[133]:


#Predicted
Std_model.predict(test.iloc[:,0:8])


# In[134]:


#actual
test.iloc[:,9]


# In[ ]:





# In[135]:


#using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
x=scale.fit_transform(x)


# In[136]:


x


# In[138]:


x=pd.DataFrame(x)


# In[139]:


x


# In[140]:


Glass_df=pd.concat([x,y],axis=1)


# In[141]:


Glass_df


# In[142]:


MinMaxmodel=knn(n_neighbors=3)
MinMaxmodel.fit(train.iloc[:,0:8],train.iloc[:,9])


# In[143]:


train_acc=np.mean(MinMaxmodel.predict(train.iloc[:,0:8])==train.iloc[:,9])
train_acc


# In[144]:


test_acc=np.mean(MinMaxmodel.predict(test.iloc[:,0:8])==test.iloc[:,9])
test_acc


# In[145]:


#predicted
MinMaxmodel.predict(test.iloc[:,0:8])


# In[146]:


#actual
test.iloc[:,9]


# In[ ]:




