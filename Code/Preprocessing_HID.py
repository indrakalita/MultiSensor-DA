#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras import layers, models, Model, optimizers
from keras.preprocessing import image
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout, Input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from skimage import io
import scipy.io
import torch


# In[2]:


#Import pavia four classes for feature extraction
pavia_data=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_data4.csv')
pavia_classes=pd.read_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_classes4.csv')
del pavia_data['Unnamed: 0']
del pavia_classes['Unnamed: 0']
print(pavia_data.shape)
print(pavia_classes.shape)

#pavia_data['classes']=pavia_classes['0']
#pavia_data.shape,pavia_classes.shape


# In[3]:


# Train autoencoder to reduce the features
# Train autoencoder
import keras
from keras import layers
from keras import regularizers

# This is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(102,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='elu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(102, activation='elu')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)


# In[4]:


from sklearn.model_selection import train_test_split
autoencoder.compile(optimizer='adam', loss='mse')

x_train, x_test = train_test_split(pavia_data, test_size=0.25, random_state=42)
# Run the model
autoencoder.fit(x_train, x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test, x_test))


# In[5]:


encoder.compile(optimizer='adam', loss='mse')
encoded_imgs = encoder.predict(pavia_data)

df=pd.DataFrame(encoded_imgs)
vhr_classes = pd.DataFrame(pavia_classes)
df.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_feature64_feature.csv')
vhr_classes.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/pavia_feature64_label.csv')
print(df.shape)


# In[3]:


df1=pavia_data[pavia_data['classes']==1]
df2=pavia_data[pavia_data['classes']==2]
df3=pavia_data[pavia_data['classes']==3]
df4=pavia_data[pavia_data['classes']==4]
del df1['classes']
del df2['classes']
del df3['classes']
del df4['classes']
df1.shape,df2.shape,df3.shape,df4.shape


# In[4]:


# Scale up the values to save it in image format

#df1 = np.multiply(df1,100000)
xx = np.amax(np.amax(df1))
df1 = df1*(255/xx)
xx = np.amax(np.amax(df1))
print(xx)

xx = np.amax(np.amax(df2))
df2 = df2*(255/xx)
xx = np.amax(np.amax(df2))
print(xx)

xx = np.amax(np.amax(df3))
df3 = df3*(255/xx)
xx = np.amax(np.amax(df3))
print(xx)

xx = np.amax(np.amax(df4))
df4 = df4*(255/xx)
xx = np.amax(np.amax(df4))
print(xx,len(df4.index))


# In[28]:


from numpy import genfromtxt
import pandas as pd
from PIL import Image
import cv2
a=[]
label = []
out_dir ='/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_data/Class1/'
k = 0
c1 = 0
while(k<len(df1.index)):
  x=np.array(df1[k:k+102])
  #print(x.shape[0])
  if(x.shape[0]==102):
    a.append(x)
    label.append(0)
    #img = str(k)+'.jpg'
    #filename = 'opencv'+str(i)+'.png'
    #x=np.resize(x,[102,102])
    #cv2.imwrite(out_dir+img, x)
    c1+=1
    print('Image of class1 is being saved',c1)
  k+=102

k=0
c2=0
print('Class 1 done and Class 2 started')
out_dir ='/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_data/Class2/'
while(k<len(df2.index)):
  x=np.array(df2[k:k+102])
  #print(x.shape)
  if(x.shape[0]==102):
    a.append(x)
    label.append(1)
    #img = str(k)+'.jpg'
    #filename = 'opencv'+str(i)+'.png'
    #x=np.resize(x,[102,102])
    #cv2.imwrite(out_dir+img, x)
    #print('[INFO] Save Change Map ...')
    #p.to_csv('drive/My Drive/csv_pavia/class1/csv'+str(int(k/102))+'.csv')
    c2+=1
    print('Image of class2 is being saved',c2)
  k+=102


k=0
c3=0
print('Class 2 done and Class 3 started')
out_dir ='/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_data/Class3/'
while(k<len(df3.index)):
  x=np.array(df3[k:k+102])
  #print(x.shape)
  if(x.shape[0]==102):
    a.append(x)
    label.append(2)
    #img = str(k)+'.jpg'
    #filename = 'opencv'+str(i)+'.png'
    #x=np.resize(x,[102,102])
    #cv2.imwrite(out_dir+img, x)
    #print('[INFO] Save Change Map ...')
    #p.to_csv('drive/My Drive/csv_pavia/class1/csv'+str(int(k/102))+'.csv')
    c3+=1
    print('Image of class3 is being saved',c3)
  k+=102


k=0
c4=0
print('Class 3 done and Class 4 started')
out_dir ='/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_data/Class4/'
while(k<len(df4.index)):
  x=np.array(df4[k:k+102])
  #print(x.shape)
  if(x.shape[0]==102):
    a.append(x)
    label.append(3)
    #img = str(k)+'.jpg'
    #filename = 'opencv'+str(i)+'.png'
    #x=np.resize(x,[102,102])
    #cv2.imwrite(out_dir+img, x)
    #print('[INFO] Save Change Map ...')
    #p.to_csv('drive/My Drive/csv_pavia/class1/csv'+str(int(k/102))+'.csv')
    c4+=1
    print('Image of class4 is being saved',c4)
  k+=102
da = np.array(a)
label = np.array(label)
print(da.shape,label.shape)
#da = np.stack(a)
#da.shape


# In[29]:


# Data preparing for the feature extraction
from sklearn.model_selection import train_test_split

# Model / data parameters
num_classes = 4
input_shape = (102, 102, 1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(da,label, test_size=0.25, random_state=42)
print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)
all_data = np.expand_dims(da, -1)
X_train1 = np.expand_dims(X_train1, -1)
X_test1 = np.expand_dims(X_test1, -1)
print(X_train1.shape, "train samples")
print(X_test1.shape, "test samples")

# convert class vectors to binary class matrices
y_train1 = to_categorical(y_train1, num_classes)
y_test1 = to_categorical(y_test1, num_classes)
print(y_train1.shape)


# In[9]:


# Creating the CNN model
def build_model(inputShape, no_classes):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    x = Conv2D(16, kernel_size=(3, 3),activation='relu')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3),activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3),activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1000, activation = 'relu',name="intermediate_layer0")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation = 'relu',name="intermediate_layer1")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(no_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


# In[18]:


IMG_SHAPE = (102, 102, 1)
BATCH_SIZE = 256
EPOCHS = 20
op = Adam(lr=0.0001)
no_classes = 4
model=build_model(IMG_SHAPE, no_classes)
model.compile(optimizer = op,loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()


# In[19]:


model.fit(X_train1, y_train1, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)


# In[20]:


# evaluating the model's performance on the test data
score = model.evaluate(X_test1, y_test1, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1]*100)


# In[30]:


# feature extraction from the model for all the pavia data
print("Feature extraction from the model")
feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="intermediate_layer1").output)
feature_extractor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(feature_extractor.summary())
pavia_data_102 = feature_extractor.predict(all_data)
pavia_data_102.shape


# In[35]:


# saving the features extracted from images of dimension 102 in csv
pavia_data_102=np.array(pavia_data_102)
label = np.array(label)
import pandas as pd
pavia_data_102=pd.DataFrame(pavia_data_102)
pavia_label=pd.DataFrame(label)
pavia_data_102.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_image64_features.csv')
pavia_label.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/Pavia_image64_label.csv')
pavia_data_102.head()

