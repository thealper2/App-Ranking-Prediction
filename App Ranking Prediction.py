#!/usr/bin/env python
# coding: utf-8

# In[266]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import dateutil
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[208]:


df = pd.read_csv("googleplaystore.csv")


# In[209]:


df.shape


# In[210]:


df.isnull().sum()


# In[211]:


df["Type"].value_counts()


# In[212]:


df["Price"].value_counts()


# In[213]:


df["Android Ver"].value_counts()


# In[214]:


df["Content Rating"].value_counts()


# # Preprocess

# In[215]:


def DateParser(text):
    try:
        date = dateutil.parser.parse(text)
        return date.strftime("%d/%m/%Y")
    
    except:
        return np.NaN


# In[216]:


def SizeParser(text):
    try:
        size = text[:len(text)-1]
        return float(size)
    
    except:
        pass


# In[217]:


def PriceParser(text):
    if text != 0:
        text = text.replace("$", "")
        return float(text)
    else:
        return float(0.0)


# In[218]:


def android_version(text):
    if not text.startswith("Varies"):
        string = str(text[:3])
        return float(string)
    else:
        return 4.4


# In[219]:


df["Last Updated"] = df["Last Updated"].apply(DateParser)
df["Last Updated"] = pd.to_datetime(df["Last Updated"])


# In[220]:


df = df.dropna(how="any")


# In[221]:


df.shape


# In[222]:


df["Size"] = df["Size"].apply(SizeParser)


# In[223]:


df["Size"].astype("float64")


# In[224]:


df = df.dropna(how="any")


# In[225]:


df.shape


# In[226]:


le = LabelEncoder()
le.fit(df["Category"])
df["Category"] = le.transform(df["Category"])


# In[227]:


le2 = LabelEncoder()
le2.fit(df["Content Rating"])
df["Content Rating"] = le2.transform(df["Content Rating"])


# In[228]:


le3 = LabelEncoder()
le3.fit(df["Type"])
df["Type"] = le3.transform(df["Type"])


# In[229]:


df["Installs"] = df["Installs"].apply(lambda x: re.sub(r'\D', '', x))
df["Installs"].astype("float64")


# In[230]:


df["Android Ver"] = df["Android Ver"].apply(android_version)


# In[231]:


df["Price"] = df["Price"].apply(PriceParser)


# In[232]:


df = df.drop(["App", "Genres", "Current Ver"], axis=1)


# In[233]:


sns.heatmap(df.corr(), annot=True, cmap="viridis")


# # Model Training

# In[256]:


X = df.drop(["Rating", "Reviews", "Last Updated", "Android Ver"], axis=1)
y = df["Rating"]


# In[257]:


X = X[:1000]
y = y[:1000]


# In[258]:


sc = StandardScaler()
X_scaled = sc.fit_transform(X_train)


# In[259]:


pca = PCA()
X_pca = pca.fit_transform(X_scaled)


# In[260]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4242)


# In[261]:


sc = StandardScaler()
pca = PCA()
rf_model = RandomForestRegressor()

model = Pipeline([
    ("sc", sc),
    ("pca", pca),
    ("rf_model", rf_model)
])


# In[262]:


model.fit(X_train, y_train)


# In[263]:


model.score(X_test, y_test)


# In[264]:


y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[265]:


acc = round(rf_model.score(X_test, y_test) * 100, 2)
acc


# In[267]:


joblib.dump(model, "RandomForest.pkl")


# In[ ]:




