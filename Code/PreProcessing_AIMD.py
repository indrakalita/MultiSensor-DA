#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras import layers, models, Model, optimizers
from keras.preprocessing import image
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import os


# In[2]:


# Data Preprocessing
from PIL import Image

path = "/home/indrajit/MultiSensor/Data/VHR"
fileA = os.listdir(path)
x1 = []
y1 = []
d={'FruitTree50':0,'IrrigatedLand50':1,'Pasture50':2,'Vineyards50':3}
for folder_name in fileA:
    files_list = os.listdir(os.path.join(path, folder_name))
    for lineA in files_list:
        x1.append(np.asarray(Image.open(os.path.join(path, folder_name, lineA)).convert('RGB').resize((224, 224))))
        y1.append(d[folder_name])
x1 = np.asarray(x1)/255.0
y1 = np.asarray(y1)
x1.shape,y1.shape


# In[3]:


# splitting the dataset into train and test for training the transfer model by freezing the convolutional layers of the model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_val, Y_train, Y_val = train_test_split(x1, y1, test_size=0.1, random_state=42)
Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))
Y_train = to_categorical(Y_train, 4)
Y_val = to_categorical(Y_val, 4)
X_train.shape,X_val.shape,Y_train.shape,Y_val.shape


# In[4]:


# Finetuning for VHR dataset
base_model = VGG16(weights='imagenet', include_top=True)
#base_model.summary()
# Freeze all convolution blocks exect the flatten layers in the network
for layer in base_model.layers[:19]:
    layer.trainable = False

# create a fully connected network to attatch to the last but one layer of the VGG16
x = base_model.layers[-2].output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(64, activation='relu')(x)
x = Dense(4, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=base_model.input, outputs=x)

# Make sure you have frozen the correct layers
#for i, layer in enumerate(transfer_model.layers):
 #   print(i, layer.name, layer.trainable)


#transfer_model.summary()
base_model.summary()


# In[65]:


# compile the model 
learning_rate= 0.0001
transfer_model.compile(loss="binary_crossentropy",
                       optimizer=optimizers.Adam(lr=learning_rate),
                       metrics=["accuracy"])


# In[66]:


history = transfer_model.fit(X_train, Y_train,
                             batch_size = 64, epochs=15,
                             validation_data=(X_val,Y_val)) #train the model


# In[67]:


# extracting the features after the fine tuning from the model and storing them as csv
model = Model(inputs=transfer_model.inputs, outputs=transfer_model.layers[-2].output)
print(model.summary())
encoder_data=model.predict(x1)

df=pd.DataFrame(encoder_data)
vhr_classes = pd.DataFrame(y1)
df.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_finetune64_features.csv')
vhr_classes.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_finetune64_label.csv')
print(df.shape)


# In[7]:


# FEature extraction from VHR without finetuning
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
model.summary()
encoder_data=model.predict(x1)

df=pd.DataFrame(encoder_data)
vhr_classes = pd.DataFrame(y1)
df.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_feature4096_features.csv')
vhr_classes.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_feature4096_label.csv')
print(df.shape)


# In[8]:


# Train autoencoder to reduce the features
# Train autoencoder
import keras
from keras import layers
from keras import regularizers

# This is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(4096,))
# "encoded" is the encoded representation of the input
x = layers.Dense(512, activation='elu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = layers.Dense(encoding_dim, activation='elu',
                activity_regularizer=regularizers.l1(10e-5))(x)
# "decoded" is the lossy reconstruction of the input
x = layers.Dense(512, activation='elu')(encoded)
decoded = layers.Dense(4096, activation='elu')(x)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)


# In[9]:


from sklearn.model_selection import train_test_split
autoencoder.compile(optimizer='adam', loss='mse')

x_train, x_test = train_test_split(encoder_data, test_size=0.25, random_state=42)
# Run the model
autoencoder.fit(x_train, x_train,epochs=5000,batch_size=256,shuffle=True,validation_data=(x_test, x_test))


# In[11]:


encoder.compile(optimizer='adam', loss='mse')
encoded_imgs = encoder.predict(encoder_data)

df=pd.DataFrame(encoded_imgs)
vhr_classes = pd.DataFrame(y1)
df.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_feature64_features.csv')
vhr_classes.to_csv('/home/indrajit/MultiSensor/Saved_features/Initial/VHR_feature64_label.csv')
print(df.shape)


# In[ ]:




