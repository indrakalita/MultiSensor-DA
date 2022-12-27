#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


# In[3]:


vhr_data=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_finetune64_features.csv')
vhr_classes=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_finetune64_label.csv')
del vhr_data['Unnamed: 0']
del vhr_classes['Unnamed: 0']
print(vhr_data.isnull().values.any())
a=vhr_data.max()
a.max()
vhr_data=vhr_data/a.max()
vhr_data.isnull().values.any()
vhr_data['classes']=vhr_classes
print(vhr_data.shape,vhr_classes.shape)

vd1=vhr_data[vhr_data['classes']==0]
vd2=vhr_data[vhr_data['classes']==1]
vd3=vhr_data[vhr_data['classes']==2]
vd4=vhr_data[vhr_data['classes']==3]
print(vd1.shape,vd2.shape,vd3.shape,vd4.shape)
#v_data = np.concatenate((vd1,vd2,vd3,vd4), axis=0)
v_classes1=vd1['classes']
del vd1['classes']
v_classes2=vd2['classes']
del vd2['classes']
v_classes3=vd3['classes']
del vd3['classes']
v_classes4=vd4['classes']
del vd4['classes']
v_data = np.concatenate((vd1,vd2,vd3,vd4), axis=0)
v_classes = np.concatenate((v_classes1,v_classes2,v_classes3,v_classes4), axis=0)

print(v_classes.shape,v_data.shape)


# In[4]:


pavia_data=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_feature64_feature.csv')
pavia_classes=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_feature64_label.csv')
del pavia_data['Unnamed: 0']
del pavia_classes['Unnamed: 0']
print(pavia_data.isnull().values.any())
a=pavia_data.max()
a.max()
pavia_data=pavia_data/a.max()
print(pavia_data.isnull().values.any())

pavia_classes=pavia_classes.replace(to_replace =[1,2,3,4], value =[0,1,2,3])
pavia_data['classes']=pavia_classes
print(pavia_data.shape,pavia_classes.shape)

pd1=pavia_data[pavia_data['classes']==0]
pd2=pavia_data[pavia_data['classes']==1]
pd3=pavia_data[pavia_data['classes']==2]
pd4=pavia_data[pavia_data['classes']==3]
print(pd1.shape,pd2.shape,pd3.shape,pd4.shape)

p_classes1=pd1['classes']
del pd1['classes']
p_classes2=pd2['classes']
del pd2['classes']
p_classes3=pd3['classes']
del pd3['classes']
p_classes4=pd4['classes']
del pd4['classes']
p_data = np.concatenate((pd1, pd2,pd3,pd4), axis=0)
p_classes = np.concatenate((p_classes1,p_classes2,p_classes3,p_classes4), axis=0)
print(p_classes.shape,p_data.shape)


# In[5]:


pavia_classes = np.array(pavia_classes)
print(np.unique(pavia_classes))


# In[53]:


merge = np.concatenate((v_data, p_data), axis=0)
mergeC = np.concatenate((v_classes, p_classes), axis=0)
#mergeC = mergeC[:,0]
print(merge.shape, mergeC.shape)


# In[5]:


# tsne plot for target (pavia) data
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(p_data)

target_names = [0, 1, 2, 3]
target_ids = range(4)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c'#, 'm', 'y', 'k', 'orange'
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[p_classes == i, 0], X_2d[p_classes == i, 1], c=c, label=label)
plt.legend()
plt.show()


# In[54]:


# Active learning using k-means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


#K-means clustering for target domain
kmeansT = KMeans(n_clusters=4, random_state=0).fit(p_data)
cluster_labels = kmeansT.fit_predict(p_data)
silhouette_avg = silhouette_score(p_data, cluster_labels)
print(silhouette_avg) # Value should be between 1 to -1 (1=best, -1 = wrost)


#K-means clustering for domain clustering
#kmeans = KMeans(n_clusters=4, random_state=0).fit(p_data)


# In[6]:


# Divide the source data into seperate class bucket based on their standard devation scores
std = []
C0 = []
C1 =[]
C2 = []
C3 = []
np.set_printoptions(suppress=True)
for i in range(len(v_data)):
    mean_std = np.std(v_data[i], axis = 0)
    if v_classes[i] == 0:
        temp = np.concatenate((i, mean_std, v_classes[i] ), axis=None)
        C0.append(temp)
    if v_classes[i] == 1:
        temp = np.concatenate((i, mean_std, v_classes[i] ), axis=None)
        C1.append(temp)
    if v_classes[i] == 2:
        temp = np.concatenate((i, mean_std, v_classes[i] ), axis=None)
        C2.append(temp)
    if v_classes[i] == 3:
        temp = np.concatenate((i, mean_std, v_classes[i] ), axis=None)
        C3.append(temp)
C0 = np.array(C0)
C1 = np.array(C1)
C2 = np.array(C2)
C3 = np.array(C3)
print(C0.shape,C1.shape,C2.shape,C3.shape)
C0_sort = C0[np.argsort(C0[:, 1])]
C1_sort = C1[np.argsort(C1[:, 1])]
C2_sort = C2[np.argsort(C2[:, 1])]
C3_sort = C3[np.argsort(C3[:, 1])]


#print(*d_sort, sep='\n')
print('C0 clusters')
print(*C0_sort[0:10], sep='\n')
print('C1 clusters')
print(*C1_sort[0:10], sep='\n')
print('C2 clusters')
print(*C2_sort[0:10], sep='\n')
print('C3 clusters')
print(*C3_sort[0:10], sep='\n')


# In[7]:


C0T = []
C1T =[]
C2T = []
C3T = []
for i in range(len(p_data)):
    mean_std = np.std(p_data[i], axis = 0)
    #print(mean_std)
    temp0 = np.absolute(mean_std - C0.std(axis = 0)[1])
    temp1 = np.absolute(mean_std - C1.std(axis = 0)[1])
    temp2 = np.absolute(mean_std - C2.std(axis = 0)[1])
    temp3 = np.absolute(mean_std - C3.std(axis = 0)[1])
    if ((temp0 < temp1) and (temp0< temp2) and (temp0< temp3)):
        temp = np.concatenate((i, mean_std, '0', p_classes[i]), axis=None)
        C0T.append(temp)
    if ((temp1 < temp0) and (temp1< temp2) and (temp1< temp3)):
        temp = np.concatenate((i, mean_std, '1', p_classes[i]), axis=None)
        C1T.append(temp)
    if ((temp2 < temp1) and (temp2< temp0) and (temp2< temp3)):
        temp = np.concatenate((i, mean_std, '2', p_classes[i]), axis=None)
        C2T.append(temp)
    if ((temp3 < temp1) and (temp3< temp2) and (temp3< temp0)):
        temp = np.concatenate((i, mean_std, '3', p_classes[i]), axis=None)
        C3T.append(temp)
        
C0T = np.array(C0T)
C1T = np.array(C1T)
C2T = np.array(C2T)
C3T = np.array(C3T)
print(C0T.shape,C1T.shape, C2T.shape, C3T.shape)
C0T_sort = C0T[np.argsort(C0T[:, 1])]
C1T_sort = C1T[np.argsort(C1T[:, 1])]
C2T_sort = C2T[np.argsort(C2T[:, 1])]
C3T_sort = C3T[np.argsort(C3T[:, 1])]


#print(*d_sort, sep='\n')
print('C0 clusters')
print(*C0T_sort[0:5], sep='\n')
print(*C0T_sort[-5:], sep='\n')
print('C1 clusters')
print(*C1T_sort[0:5], sep='\n')
print(*C1T_sort[-5:], sep='\n')
print('C2 clusters')
print(*C2T_sort[0:5], sep='\n')
print(*C2T_sort[-5:], sep='\n')
print('C3 clusters')
print(*C3T_sort[0:5], sep='\n')
print(*C3T_sort[-5:], sep='\n')


# In[56]:


sample = 3 # Put the value half of the active sample
x =[]
for i in range(sample):
    val = C0T_sort[i][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C1T_sort[i][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C2T_sort[i][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C3T_sort[i][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    
    val = C0T_sort[len(C0T_sort)-(1+i)][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C1T_sort[len(C1T_sort)-(1+i)][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C2T_sort[len(C2T_sort)-(1+i)][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    val = C3T_sort[len(C3T_sort)-(1+i)][0]
    val = val.astype(np.int)
    #print(val)
    x.append(val)
    
x = np.asarray(x)
print(x)
#print(x[1])
#x[2] = 2195
#x[3] = 109445
#x[4] = 38411
#x[8] = 773
print(x)


# In[57]:


print(np.unique(C1T[:,3],return_counts=True))
print(np.unique(C3T[:,3],return_counts=True))


# In[58]:


p_data_train = np.concatenate((p_data[x[0]:x[0]+1],p_data[x[1]:x[1]+1],
                               p_data[x[2]:x[2]+1],p_data[x[3]:x[3]+1]), axis=0)



p_classes_train = np.concatenate((p_classes[x[0]:x[0]+1],p_classes[x[1]:x[1]+1],
                                  p_classes[x[2]:x[2]+1],p_classes[x[3]:x[3]+1]), axis=0)


#p_data_train = np.concatenate((p_data[x[0]:x[0]+1],p_data[x[1]:x[1]+1],
#                               p_data[x[2]:x[2]+1],p_data[x[3]:x[3]+1],
#                               p_data[x[4]:x[4]+1],p_data[x[5]:x[5]+1],
#                               p_data[x[6]:x[6]+1],p_data[x[7]:x[7]+1]), axis=0)



#p_classes_train = np.concatenate((p_classes[x[0]:x[0]+1],p_classes[x[1]:x[1]+1],
#                                  p_classes[x[2]:x[2]+1],p_classes[x[3]:x[3]+1],
#                                  p_classes[x[4]:x[4]+1],p_classes[x[5]:x[5]+1],
#                                  p_classes[x[6]:x[6]+1],p_classes[x[7]:x[7]+1]), axis=0)



#p_data_train = np.concatenate((p_data[x[0]:x[0]+1],p_data[x[1]:x[1]+1],
#                               p_data[x[2]:x[2]+1],p_data[x[3]:x[3]+1],
#                              p_data[x[4]:x[4]+1],p_data[x[5]:x[5]+1],
#                              p_data[x[6]:x[6]+1],p_data[x[7]:x[7]+1],
#                              p_data[x[8]:x[8]+1],p_data[x[9]:x[9]+1],
#                              p_data[x[10]:x[10]+1],p_data[x[11]:x[11]+1]), axis=0)



#p_classes_train = np.concatenate((p_classes[x[0]:x[0]+1],p_classes[x[1]:x[1]+1],
#                                  p_classes[x[2]:x[2]+1],p_classes[x[3]:x[3]+1],
#                                  p_classes[x[4]:x[4]+1],p_classes[x[5]:x[5]+1],
#                                 p_classes[x[6]:x[6]+1],p_classes[x[7]:x[7]+1],
#                                  p_classes[x[8]:x[8]+1],p_classes[x[9]:x[9]+1],
#                                  p_classes[x[10]:x[10]+1],p_classes[x[11]:x[11]+1]), axis=0)


#p_data_train = np.concatenate((p_data[x[0]:x[0]+1],p_data[x[1]:x[1]+1],
#                               p_data[x[2]:x[2]+1],p_data[x[3]:x[3]+1],
#                               p_data[x[4]:x[4]+1],p_data[x[5]:x[5]+1],
#                               p_data[x[6]:x[6]+1],p_data[x[7]:x[7]+1],
#                               p_data[x[8]:x[8]+1],p_data[x[9]:x[9]+1],
#                               p_data[x[10]:x[10]+1],p_data[x[11]:x[11]+1],
#                               p_data[x[12]:x[12]+1],p_data[x[13]:x[13]+1],
#                               p_data[x[14]:x[14]+1],p_data[x[15]:x[15]+1]), axis=0)



#p_classes_train = np.concatenate((p_classes[x[0]:x[0]+1],p_classes[x[1]:x[1]+1],
#                                  p_classes[x[2]:x[2]+1],p_classes[x[3]:x[3]+1],
#                                  p_classes[x[4]:x[4]+1],p_classes[x[5]:x[5]+1],
#                                  p_classes[x[6]:x[6]+1],p_classes[x[7]:x[7]+1],
#                                  p_classes[x[8]:x[8]+1],p_classes[x[9]:x[9]+1],
#                                  p_classes[x[10]:x[10]+1],p_classes[x[11]:x[11]+1],
#                                  p_classes[x[12]:x[12]+1],p_classes[x[13]:x[13]+1],
#                                  p_classes[x[14]:x[14]+1],p_classes[x[15]:x[15]+1]), axis=0)




#p_data_train = np.concatenate((p_data[x[0]:x[0]+1],p_data[x[1]:x[1]+1],
#                               p_data[x[2]:x[2]+1],p_data[x[3]:x[3]+1],
#                              p_data[x[4]:x[4]+1],p_data[x[5]:x[5]+1],
#                               p_data[x[6]:x[6]+1],p_data[x[7]:x[7]+1],
#                               p_data[x[8]:x[8]+1],p_data[x[9]:x[9]+1],
#                               p_data[x[10]:x[10]+1],p_data[x[11]:x[11]+1],
#                               p_data[x[12]:x[12]+1],p_data[x[13]:x[13]+1],
#                               p_data[x[14]:x[14]+1],p_data[x[15]:x[15]+1],
#                               p_data[x[16]:x[16]+1],p_data[x[17]:x[17]+1],
#                               p_data[x[18]:x[18]+1],p_data[x[19]:x[19]+1]), axis=0)



#p_classes_train = np.concatenate((p_classes[x[0]:x[0]+1],p_classes[x[1]:x[1]+1],
#                                  p_classes[x[2]:x[2]+1],p_classes[x[3]:x[3]+1],
#                                  p_classes[x[4]:x[4]+1],p_classes[x[5]:x[5]+1],
#                                  p_classes[x[6]:x[6]+1],p_classes[x[7]:x[7]+1],
#                                  p_classes[x[8]:x[8]+1],p_classes[x[9]:x[9]+1],
#                                  p_classes[x[10]:x[10]+1],p_classes[x[11]:x[11]+1],
#                                  p_classes[x[12]:x[12]+1],p_classes[x[13]:x[13]+1],
#                                  p_classes[x[14]:x[14]+1],p_classes[x[15]:x[15]+1],
#                                  p_classes[x[16]:x[16]+1],p_classes[x[17]:x[17]+1],
#                                  p_classes[x[18]:x[18]+1],p_classes[x[19]:x[19]+1]), axis=0)


print(p_data_train.shape,p_classes_train.shape)
p_data_test = p_data 
p_classes_test = p_classes 
print(p_classes_test.shape,p_data_test.shape)
print(p_classes_train)


# In[59]:


# Run if VHR to pavia
print(v_data.shape,v_classes.shape)


# In[60]:


import collections
elements_count = collections.Counter(v_classes)
for key, value in elements_count.items():
   print(f"{key}: {value}")


# In[61]:


# Merge the data. Here put one type of data in both the cases to make it unsupervised
merge = np.concatenate((v_data, p_data_train), axis=0)  #replace pavia_data with vhr_data if vhr is the source
print(merge.shape)
mergeC = np.concatenate((v_classes, p_classes_train), axis=0)
print(mergeC.shape)


# In[62]:


import collections
elements_count = collections.Counter(mergeC)
for key, value in elements_count.items():
   print(f"{key}: {value}")


# In[63]:


from keras.utils import to_categorical

#print(v_classes.shape,p_classes.shape)
mergeC = to_categorical(mergeC, 4)

p_classes_test = to_categorical(p_classes_test, 4)
#v_classes.shape,p_classes.shape
p_classes_train = to_categorical(p_classes_train, 4)

print(mergeC.shape,p_classes_test.shape,p_classes_train.shape)


# In[64]:


# MLP on features extracted for both datasets with dimension 64

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout

# define the keras model
model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dropout(0.5)),
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5)),
model.add(Dense(4, activation='softmax'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.summary()


# In[65]:


model.fit(merge,mergeC, epochs=100, batch_size=64)
# evaluate the keras model
# _, accuracy = model.evaluate(vhr_data, v_classes)
# print('Accuracy: %.2f' % (accuracy*100))


# In[66]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
xx = np.argmax(p_classes_test, axis=1)
pred = model.predict(p_data_test)
yy = np.argmax(pred, axis=1)
print(xx.shape,yy.shape)
print(pred)


# In[67]:


print(classification_report(xx,yy))
_, accuracy = model.evaluate(p_data_test, p_classes_test)
print('Accuracy_P: %.2f' % (accuracy*100))

