#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from sklearn import metrics
from sklearn.decomposition import PCA


# In[23]:


train_X = pd.read_csv("train.csv",header=None, names=["name", "num"])
print(train_X.shape)
print(train_X.name[0])
y = train_X.num
print(y)


# In[29]:


train_X = pd.read_csv("example1.csv",header=None)
train_X[0][0]


# In[32]:


face_cascade = cv2.CascadeClassifier(r'C:\Users\user\AppData\Local\conda\conda\pkgs\libopencv-3.4.2-h20b85fd_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

 
face_filename = 1
def detect(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=3,)
    #print(faces)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        name = "ii"+ str(filename)
        print(name)
        #cv2.imwrite(name, f)
        
    print('Working with %s' % filename)
for i in range(1):
    detect(train_X[0][i])


# In[ ]:





# In[ ]:




