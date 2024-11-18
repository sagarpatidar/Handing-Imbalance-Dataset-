#!/usr/bin/env python
# coding: utf-8

# # Imbalanced Dataset Handling 

# 1.Upsamping
# 2.Downsampling

# In[2]:


import pandas as pd 
import numpy as np


# In[6]:


#Creating a data points Using Numpy
np.random.seed(123)
n_samples =1000
class_0_ratio = 0.9
n_class_0 = int (n_samples*class_0_ratio)
n_class_1 = n_samples -n_class_0


# In[8]:


n_class_0,n_class_1


# In[15]:


class_0 =pd.DataFrame({
    'feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
    'feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
    'target':[0]*n_class_0
})
class_1 =pd.DataFrame({
    'feature_1':np.random.normal(loc=2,scale=1,size=n_class_1),
    'feature_2':np.random.normal(loc=2,scale=1,size=n_class_1),
    'target':[1]*n_class_1
})


# In[19]:


df= pd.concat([class_0,class_1]).reset_index(drop=True)


# In[20]:


df.head


# In[23]:


df['target'].value_counts()


# # upsampling 
# 
# 

# In[33]:


df_minority= df[df['target']==1]
df_majority=df[df['target']==0]


# In[41]:


df_minority.head()


# In[35]:


df_majority.head()


# In[37]:


from sklearn.utils import resample 


# In[55]:


df_manority_upsample=resample(df_minority,
                               replace=True,
                               n_samples=len(df_majority),
                                random_state=42)


# In[67]:


df_manority_upsample.shape


# In[76]:


df_manority_upsample.value_counts()


# In[80]:


df_manority_upsample['target'].value_counts()


# In[90]:


df_upsampled = pd.concat([df_majority,df_manority_upsample])


# In[91]:


df_upsampled['target'].value_counts()


# In[96]:


df_upsampled.shape


# # DownSampling

# In[97]:


class_0 =pd.DataFrame({
    'feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
    'feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
    'target':[0]*n_class_0
})
class_1 =pd.DataFrame({
    'feature_1':np.random.normal(loc=2,scale=1,size=n_class_1),
    'feature_2':np.random.normal(loc=2,scale=1,size=n_class_1),
    'target':[1]*n_class_1
})


# In[98]:


df= pd.concat([class_0,class_1]).reset_index(drop=True)


# In[124]:


df_minority=df[df['target']==1]
df_majority=df[df['target']==0]


# In[132]:


df_majority_downsample=resample(df_majority,
                               replace=False,
                               n_samples=len(df_minority),
                                random_state=42)


# In[134]:


df_majority_downsample.shape


# In[153]:


df_downsample=pd.concat([df_minority , df_majority_downsample])


# In[154]:


df_downsample['target'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:




