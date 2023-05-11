#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import wbgapi as wb
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[35]:


def data(x,y,z):
  """Preprocessing data and returning original, processed data"""
  data = wb.data.DataFrame(x, mrv = y)
  data1 = pd.DataFrame(data.sum())
  data1.columns = z
  data2 = data1.rename_axis("year")
  return data, data2
def string(x):
  """Converting Year column to Integers """
  # print(x)
  l = []
  z = []
  for i in x.index:
      c = i.split("YR")
      z.append(int(c[1]))
  # print(z)
  x["year"] = z
  return x


# In[3]:


indicator = ["EN.ATM.GHGT.ZG","EN.ATM.NOXE.ZG"] # indicators


# In[4]:


data_ghg_O, data_ghg_R = data(indicator[0],30, ["Greenhouse gases"]) #greenhouse gases data


# In[5]:


new_data_GHG = string(data_ghg_R) #calling string function to convert year column to integer


# In[6]:


data_N2O_O, data_N2O_R = data(indicator[1], 30, ["N2O"]) #N2O data


# In[7]:


new_data_N2O = string(data_N2O_R) #calling string function to convert year column to intege


# In[8]:


def exp_growth(t, scale, growth):
    """Computes exponential function with scale and growth as free parameters"""
    f = scale * np.exp(growth * (t-1990))
    return f


# In[9]:


popr, pcov = curve_fit(exp_growth, data_ghg_R["year"], data_ghg_R["Greenhouse gases"]) #calling curve_fit


# In[10]:


#plotting graph between data and curve_fit
data_ghg_R["pop_exp"] = exp_growth(data_ghg_R["year"], *popr)
plt.figure()
plt.plot(data_ghg_R["year"], data_ghg_R["Greenhouse gases"], label="Greenhouse gases")
plt.plot(data_ghg_R["year"], data_ghg_R["pop_exp"], label="fit")
plt.legend()
plt.title("Curve Fit and data line of Greenhouse Gases")
plt.xlabel("year")
plt.ylabel("Greenhouse gases")
plt.show()
print()


# In[11]:


def err_ranges(x, func, param, sigma):
   """
   Calculates the upper and lower limits for the function, parameters and
   sigmas for single value or array x. Functions values are calculated for
   all combinations of +/- sigma and the minimum and maximum is determined.
   Can be used for all number of parameters and sigmas >=1.
   This routine can be used in assignment programs.
   """
   import itertools as iter
    
   # initiate arrays for lower and upper limits

   lower = func(x, *param)
   upper = lower

   uplow = [] # list to hold upper and lower limits for parameters
   for p,s in zip(param, sigma):
       pmin = p - s
       pmax = p + s
       uplow.append((pmin, pmax))
        
   pmix = list(iter.product(*uplow))

   for p in pmix:
       y = func(x, *p)
       lower = np.minimum(lower, y)
       upper = np.maximum(upper, y)
    
   return lower, upper


# In[17]:


#plotting graph between confidence ranges and fit data
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data_ghg_R["year"], exp_growth , popr, sigma)
plt.figure()
plt.title("exp_growth function")
plt.plot(data_ghg_R["year"], data_ghg_R["Greenhouse gases"], label="Greenhouse")
plt.plot(data_ghg_R["year"], data_ghg_R["pop_exp"], label="fit")
plt.fill_between(data_ghg_R["year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("Greenhouse gases")
plt.show()


# In[13]:


popr2, pcov2 = curve_fit(exp_growth, data_N2O_R["year"], data_N2O_R['N2O']) #curve_fit for N2O


# In[15]:


#plotting graph between N2O data and curve_fit
data_N2O_R["pop_exp"] = exp_growth(data_N2O_R["year"], *popr2)
plt.figure()
plt.plot(data_N2O_R["year"], data_N2O_R["N2O"], label="N2O")
plt.plot(data_N2O_R["year"], data_N2O_R["pop_exp"], label="fit")
plt.legend()
plt.title("Curve fit and data line of N2O")
plt.xlabel("year")
plt.ylabel("N2O")
plt.show()
print()


# In[16]:


#plotting graph between confidence ranges and fit data
sigma = np.sqrt(np.diag(pcov2))
low, up = err_ranges(data_N2O_R["year"], exp_growth , popr2, sigma)
plt.figure()
plt.title("exp_growth function")
plt.plot(data_N2O_R["year"], data_N2O_R["N2O"], label="N2O")
plt.plot(data_N2O_R["year"], data_N2O_R["pop_exp"], label="fit")
plt.fill_between(data_N2O_R["year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("N2O")
plt.show()


# In[20]:


#prepossing data for clustering
data_ghg = pd.DataFrame(data_ghg_O.iloc[:,-1])
data_N2O = pd.DataFrame(data_N2O_O.iloc[:,-1])


# In[21]:


data_ghg.columns = ["Greenhouse gases"]
data_N2O.columns = ["N2O"]


# In[22]:


data_ghg["N2O"] = data_N2O["N2O"]


# In[23]:


data_ghg_C = data_ghg.rename_axis("countries")


# In[24]:


final_data = data_ghg_C.dropna()


# In[27]:


# Visualizing data with Scatter plot
fig = plt.figure(figsize = (8, 6))
sns.scatterplot(data=final_data, x="N2O", y="Greenhouse gases", color = 'green')
plt.title("scatter plot before clustering")


# In[33]:


#plotting scatter plot for kmeans clustering
X = final_data[['N2O', 'Greenhouse gases']].copy()
kmeanModel = KMeans(n_clusters=3) # chosed 3 clusters
identified = kmeanModel.fit_predict(final_data[['N2O', 'Greenhouse gases']])
cluster_centers = kmeanModel.cluster_centers_ #getting cluster center points
#Getting unique labels
u_labels = np.unique(identified) # getting unique cluster labels
clusters_with_data = final_data[['N2O', 'Greenhouse gases']].copy()
clusters_with_data['Clusters'] = identified #add cluster column
fig = plt.figure(figsize = (10, 8))
# ploting data points
plt.scatter(clusters_with_data['N2O'],clusters_with_data['Greenhouse gases'],
c=clusters_with_data['Clusters'], cmap='viridis')
# ploting center points
#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, alpha=0.8);
plt.title("Scatter plot after clusters")
plt.xlabel('N2O')
plt.ylabel('Greenhouse gases')
plt.show()


# In[ ]:




