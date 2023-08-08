#%%
#libraries importations
import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import keras
from keras import layers

#%%
#folder locations
main_datasets = '/Home/makanji/ChickenProject/Turkey_Frames/'
home_folder =  '/Home/makanji/Master_Thesis/Datasets/'

# define the main parameters
#Encoder_Input_Dim = (800,1200,1)
#Encoder_Ouptu_Dim = (75,50,64)
image_shape = ( 800, 1200)
height = 1200
width = 800
color_channel = 3
#%%
#importation of my datasets and preprocessing
images_name = os.listdir(main_datasets)
image_shape = ( 800, 1200)
#Encoder_Input_Dim = (800,1200,1)
len(images_name)
#img_sample = images_name[0:2]
datasets = []
for i in range(len(images_name)):
    print(i)
    img = cv.imread(os.path.join(main_datasets, images_name[i]),1)
    #print('imported image shape'+ str(img.shape))
    img = cv.resize(img, dsize=(height,width))
    #print('Image reshaped to' + str(img.shape))
    datasets.append(img)
datasets = np.asarray(datasets)
print(datasets.shape)
plt.imshow(datasets[0])

#%%

#%%#
#image formatting
data = datasets.copy()
datasets.shape
data = data.astype('float32')
data = data/255.0
print(np.max(data[0]))
#%%
#splitting data into different sets
#np.random.shuffle(data)
#id = np.arange(len(data))               # Vector of dataset samples idx
#id_train = int(len(id) * 0.8)              # Train 80%
#id_valid = int(len(id) * (0.8 + 0.05))     # Valid 5%, Test 15%
#train_set, val_set, test_set = np.split(id, (id_train, id_valid))

np.random.shuffle(data)
id = np.arange(len(data))               # Vector of dataset samples idx
id_train = int(len(id) * 0.8)              # Train 80%
id_test = int(len(id) * (0.8 + 0.05))     # Valid 15%, Test 5%
train_set, test_set, val_set = np.split(id, (id_train, id_test))

# Indexing dataset subsets
dataset_train = data[train_set,:,:,:]            # Train set
dataset_valid = data[val_set,:,:,:]        # Valid set
dataset_test = data[test_set,:,:,:]            #test set

data[train_set,:,:,:].shape
len(data[train_set,:,:,:])
print(dataset_train.shape)
print(dataset_test.shape)
print(dataset_valid.shape)

#ref https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros

#%%
#image_shape = ( 800, 1200)
#model definitions and autoencoder preparations
input_img = keras.Input(shape = (image_shape[0], image_shape[1], 1))
x1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x2 = layers.MaxPooling2D((2, 2), padding='same')(x1) #st
x3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = layers.MaxPooling2D((2, 2), padding='same')(x3) #2nd
x5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = layers.MaxPooling2D((2, 2), padding='same')(x5) #3rd
x7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x6)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x7) #4th# #defining autoencoder
encoder = keras.Model(input_img, encoded)
#upscaling
x8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(encoded)
x9 = layers.UpSampling2D((2,2))(x8)
x10 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x9)
x11 = layers.UpSampling2D((2,2))(x10)
x12 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x11)
x13 = layers.UpSampling2D((2,2))(x12)
x14 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x13)
x15 = layers.UpSampling2D((2,2))(x14)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x15)
decoded.shape
x14.shape

autoencoder = keras.Model(input_img, decoded)
#%%
autoencoder.summary()
#%%
encoder.summary()
#fitting model out
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#%%
#this too heavy going down on sample sizes
train_history = autoencoder.fit(dataset_train, dataset_train,
                epochs= 50, batch_size=1,
                shuffle=True, validation_data=(dataset_valid, dataset_valid))
#have to reshape image to show color channel in input using dataset_train = np.reshape(dataset_train, (len(dataset_train), height, width, colorchannel
#%%
datasets_train_copy =dataset_train[0:200, :,:]
datasets_train_copy[0]
dataset_val_copy = dataset_valid[0:50,:,:]
print(dataset_val_copy.shape)
print(datasets_train_copy.shape)
print(datasets.shape)







#%%
autoencoder.fit(datasets_train_copy, datasets_train_copy, epochs= 2, batch_size=1, shuffle=True, validation_data=(dataset_val_copy, dataset_val_copy))

autoencoder.save_model("CAE.h5")
encoder.save("Encoder.h5")


#%%
#plotting the performance of the training
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', ('validation ' )], loc='upper left')
plt.show()
plt.savefig(('CAE_training' + '.png' ))
plt.close()






#%%

test_sample = dataset_train[0]
test_sample = test_sample.reshape(1, 1200,800,3)
test_sample.shape



reconstructed_image = autoencoder.predict(test_sample)
plt.imshow(test_sample[0,:,:,:])
plt.imshow(reconstructed_image[0,:,:,:])



#%%
#model testing for performance
#exploring the encoded image part
test_sample = dataset_test[0]
test_sample.shape
test_sample = test_sample.reshape(1,1200, 800, 3)

reconstructed_image = autoencoder.predict(test_sample)

reconstructed_image.shape
plt.imshow(reconstructed_image[0, :, :, :])
autoencoder.summary()

#encoder versions
encoder= keras.Model(input_img, encoded)
encoder.summary(())
encoded_img = encoder.predict(test_sample)
encoded_img.shape
plt.imshow(encoded_img[0, :, :, 63])