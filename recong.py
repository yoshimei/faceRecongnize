#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from sklearn import metrics
from sklearn.decomposition import PCA


# In[5]:


y = pd.read_csv("y1230.csv",header=None)
y = y[1]
y.head()


# In[12]:


urList = pd.read_csv("forgray.csv",header=None)
img = cv2.imread(urList[0][1])
print("i"+urList[0][1])
#aa = np.array(img).reshape(120000)
newList = []
for i in range(1825):
    img = cv2.imread(urList[0][i])
    
    aa = np.array(img).reshape(1,120000)
    
    aa.reshape(1,-1)
    aa = pd.DataFrame(aa)
    print(urList[0][i])
    print(aa)
    aa.to_csv('featureTrain3.csv', mode='a', header=False)


# In[22]:


X = pd.read_csv("featureTrain3.csv",header=None)


# In[23]:


X = X.drop(X[0], axis=1)
X.head


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,random_state=33,stratify = y)


# In[27]:


urList = pd.read_csv("example2.csv",header=None)
img = cv2.imread(urList[0][1])
print(urList[0][1])
newList = []
for i in range(100):
    img = cv2.imread(urList[0][i]).reshape(1,120000)
    
    aa = np.array(img).reshape(1,-1)
    #aa.reshape(1,-1)
    aa = pd.DataFrame(aa)
    print(urList[0][i])
    aa.to_csv('featureTest3.csv', mode='a', header=False)


# In[25]:


from sklearn.svm import SVC
import time
tStart = time.time()

svclassifier = SVC(kernel='linear')
print("e")
svclassifier.fit(X_train, y_train)
print("f")
y_pred = svclassifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
tEnd = time.time() #計時結束
print("Total time= %f seconds" % (tEnd - tStart))


# In[28]:


yy = pd.read_csv("featureTest3.csv",header=None)
yy = yy.drop(yy[0], axis=1)
yy.head


# In[29]:


pre = svclassifier.predict(yy)
print(pre)


# In[55]:


su = pd.read_csv("example.csv")
data = pd.DataFrame({'Id':su.Id ,'Category':pre})
data.to_csv("ans123007.csv",index=False,sep=',')
print(data)


# In[48]:


import time
tStart = time.time()
model = cv2.face.EigenFaceRecognizer_create()
model.train(np.array(X), np.array(y))

tEnd = time.time() #計時結束
print("Total time= %f seconds" % (tEnd - tStart))


# In[43]:


print(yy[0:1])


# In[49]:


for i in range(0,101):
    params = model.predict(np.array(yy[i:i+1]))
    print(params)


# In[31]:


from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV
tStart = time.time()
rf0 = RandomForestClassifier(oob_score=True,max_depth=100,max_leaf_nodes=60,n_estimators=110,max_features=8, 
                             min_samples_leaf=2, min_samples_split=4, random_state=55,n_jobs=1,bootstrap=True)
rf0.fit(X,y)
print(rf0.oob_score_)


# In[32]:


pre = rf0.predict(yy)
tEnd = time.time()


# In[33]:


print("Total time= %f seconds" % (tEnd - tStart))


# In[37]:


from sklearn import tree
tStart = time.time()
deTree= tree.DecisionTreeClassifier()
deTreeC = deTree.fit(X_train, y_train)

# 預測
test_y_predicted = deTreeC.predict(X_test)


# In[38]:


print(metrics.accuracy_score(y_test, test_y_predicted))

# 標準答案
print(deTreeC.predict(yy))
tEnd = time.time()
print("Total time= %f seconds" % (tEnd - tStart))


# In[39]:


su = pd.read_csv("example.csv")
data = pd.DataFrame({'Id':su.Id ,'Category':pre})
data.to_csv("ans123009.csv",index=False,sep=',')
print(data)


# In[ ]:




