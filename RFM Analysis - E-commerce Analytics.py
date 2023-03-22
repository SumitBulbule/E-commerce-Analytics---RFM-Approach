#!/usr/bin/env python
# coding: utf-8

# # #Problem: Dataset contains 2 years of e-commerce transactions. Data contains date time of sale, customer shipping location, price of single unit from 2016 to 2017
# 
# # #Objective: Analyze customers using RFM and provide brief details based on monetary value, frequency of buy

# In[ ]:





# ###   Importing packages

# In[1]:


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, date


# 

# ###  Importing Dataset

# In[2]:


dataset = pd.read_csv(r"C:\Users\sumit\Downloads\E-com_Data.csv")
dataset.head()


# 

# # Data Cleaning

# In[3]:


# Renaming column names


# In[4]:


dataset = dataset.rename(columns={'InvoieNo' : 'Invoice No', 'Date of purchase' : 'Date'})


# In[5]:


dataset.head(2)


# In[6]:


dataset.info()


# In[7]:


# Checking Null values


# In[8]:


dataset.isnull().sum() / len(dataset) * 100


# In[9]:


# Checking Unique and Duplicated values


# In[10]:


dataset['CustomerID'].nunique()


# In[11]:


dataset.duplicated().sum()


# In[12]:


dataset.loc[dataset.duplicated(),:]


# In[13]:


# Removing duplicates


# In[14]:


dataset = dataset.drop_duplicates(ignore_index=True)


# In[15]:


dataset.duplicated().sum()


# In[16]:


dataset.info()


# In[17]:


# dropping customer ID and missing data


# In[18]:


dataset = dataset.dropna(subset=['CustomerID'])


# In[19]:


dataset.isnull().sum()


# In[20]:


# selecting only those columns which are needed


# In[21]:


dataset.columns


# In[22]:


dataset = dataset [['CustomerID','Invoice No', 'Date', 'Price']]


# In[23]:


dataset.head()


# In[24]:


dataset.info()


# In[25]:


# converting date into datetime


# In[26]:


dataset['Date'] = pd.to_datetime(dataset['Date'])


# In[27]:


dataset['Date'].describe()


# In[28]:


import datetime as dt


# In[29]:


Latest_date = dt.datetime(2017,12,20)


# In[30]:


Latest_date


# In[31]:


# Recency = Latest_date - Date per customer id
# Frequency = Total Count (Invoice No) per customer id
# Monetory = Total sum of price per customer id


# In[32]:


RFMScore = dataset.groupby('CustomerID').agg({'Date' : lambda x : (Latest_date - x.max()).days,
                                             'Invoice No' : lambda x : x.count(),
                                             'Price' : lambda x : x.sum()})
RFMScore.rename(columns = {'Date' : 'Recency','Invoice No' : 'Frequency', 'Price' : 'Monetory'}, inplace = True)


# In[33]:


# resetting index


# In[34]:


RFMScore.reset_index().head(10)


# In[35]:


RFMScore.Recency.describe()


# In[36]:


# low recency means customers are engaged,active more whereas higher recency means customers are not actively pursuing. 


# In[37]:


RFMScore.Frequency.describe()


# In[38]:


# High Frequency means customers are engaged,active more whereas low frequency means customers are not actively pursuing.


# In[39]:


RFMScore.Monetory.describe()


# In[40]:


# High Monetory means customers are engaged,active more whereas low Montetory means customers are not actively pursuing.


# In[41]:


# Quantile format


# In[42]:


quantile = RFMScore.quantile(q=[0.25,0.50,0.75])
quantile = quantile.to_dict()
quantile


# In[43]:


# calculating recency,frequency and monetory score


# In[44]:


def Rscore(x,p,d):
    if x<= d[p][0.25]:
        return 1
    elif x<= d[p][0.50]:
        return 2
    elif x<= d[p][0.75]:
        return 3
    else:
        return 4
    
    
    
def FnMScore(x,p,d):
    if x<= d[p][0.25]:
        return 4
    elif x<= d[p][0.50]:
        return 3
    elif x<= d[p][0.75]:
        return 2
    else:
        return 1


# In[45]:


# adding new columns based on customer quantiles for recency, frequency and monetory


# In[46]:


RFMScore['R'] = RFMScore['Recency'].apply(Rscore, args=('Recency',quantile))
RFMScore['F'] = RFMScore['Frequency'].apply(FnMScore, args=('Frequency',quantile))
RFMScore['M'] = RFMScore['Monetory'].apply(FnMScore, args=('Monetory',quantile))


# In[47]:


RFMScore.head(10)


# In[48]:


RFMScore['RFMGroup'] = RFMScore.R.map(str) + RFMScore.F.map(str) + RFMScore.M.map(str)


# In[49]:


RFMScore.reset_index()


# In[50]:


RFMScore['RFMsum'] = RFMScore[['R','F','M']].sum(axis=1)


# In[51]:


RFMScore.reset_index()


# In[52]:


# < 5 Prime Customers, 5 - 8 : Normal customers > 8 : irregular customers


# In[53]:


RFMScore['RFMsum'].value_counts()


# In[54]:


RFMScore.to_csv('Manual_analysis.csv')


# In[55]:


# Assigning loyalty level to customers based on RFM


# In[56]:


Loyalty_Level = ['Gold','Platinum','Silver','Titanium']
score_cuts = pd.qcut(RFMScore.RFMsum, q=4, labels=Loyalty_Level)
RFMScore['Loyalty_Level'] = score_cuts.values
RFMScore = RFMScore.reset_index()
RFMScore


# In[ ]:





# # Building Cluster model

# In[57]:


# Creating New Database


# In[58]:


RFMScore1 = RFMScore.copy()


# In[59]:


RFMScore1.head()


# In[60]:


# Removing unwanted columns from dataset


# In[61]:


RFMScore1 = RFMScore1.iloc[:,1:4]


# In[62]:


RFMScore1.head()


# In[63]:


RFMScore1.shape


# # Visualization method

# In[64]:


RFMScore.head()


# In[65]:


final_report = RFMScore.groupby('Loyalty_Level')[['Recency','Frequency','Monetory','CustomerID']].agg({'Recency' : 'mean','Frequency' : 'mean','Monetory' : 'mean','CustomerID' : 'nunique'}).reset_index


# In[66]:


final_report


# In[68]:


# feature scaling is required since we are using euclidean distance


# In[69]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler ()
sc_data = sc.fit_transform(RFMScore1)
sc_data


# In[70]:


sc_data = pd.DataFrame(sc_data, index = RFMScore1.index, columns = RFMScore1.columns)


# In[71]:


sc_data


# In[72]:


# KMeans Clustering


# In[73]:


from sklearn.cluster import KMeans
wcss = []

for i in range (1,20):
    Kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=1000,random_state=101)
    Kmeans.fit(sc_data)
    wcss.append(Kmeans.inertia_)


# In[74]:


wcss


# In[75]:


abc = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


# In[76]:


abc


# In[77]:


# checking for clusters


# In[78]:


sns.pointplot(abc, wcss)


# In[79]:


# k = 3, 6, 7


# In[80]:


Kmeans = KMeans(n_clusters=7, init='k-means++',max_iter=1000,random_state=101)
y_Kmeans = Kmeans.fit(sc_data)
y_Kmeans


# In[81]:


RFMScore1['Cluster'] = Kmeans.labels_


# In[82]:


RFMScore1


# In[83]:


# days < 20, frequency > 70 and monetary > 150000 will consider as 0 cluster and likewise for other clusters as per need


# In[84]:


RFMScore1.to_csv('cluster.csv')


# In[85]:


dataset.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




