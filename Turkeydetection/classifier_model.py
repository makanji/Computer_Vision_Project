# library importations
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import  turkey_datasets
from turkey_datasets import x_array_1, y_array_1
print(y_array_1.shape)
#%%
#parameters definitions
Encoder_Input_Dim = (800, 1200, 1)
Encoder_Ouptu_Dim = (50, 75, 64)
color_channel = 1
classifier_Output_Dim = ( 400, 600, 4)

#%%
#dataset importation
x_array = np.load('/Home/makanji/Master_Thesis/Datasets/Turkey_arr/x_array_recent.npy')
y_array = np.load('/Home/makanji/Master_Thesis/Datasets/Turkey_arr/y_array_recent.npy')
print(x_array.shape)
print(y_array.shape)
#splitting into respective groups
#for x_arrays
from sklearn.utils import shuffle
x_array_1, y_array_1 = shuffle(x_array, y_array)
id = np.arange(len(x_array_1))
id_train = int(len(id) * 0.8)              # Train 80%
id_test = int(len(id) * (0.8 + 0.05))     # Valid 15%, Test 5%
train_set, test_set, val_set = np.split(id, (id_train, id_test))
# Indexing dataset subsets
x_train = x_array_1[train_set, :, :]              # Train set
x_valid = x_array_1[val_set, :, :]              # Valid set
x_test = x_array_1[test_set, :, :]
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
y_train = y_array_1[train_set, :, :, :]              # Train set
y_valid = y_array_1[val_set, :, :, :]              # Valid set
y_test = y_array_1[test_set, :, :, :]
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

# load the encoder
from tensorflow.keras.models import load_model
loadedEncoder = load_model("Turkey_Encoder.h5", compile =False)

#%%
#creating coded image version
encoded_img_xtrain = loadedEncoder.predict(x_train)
print(encoded_img_xtrain.shape)
encoded_img_vals = loadedEncoder.predict(x_valid)
print(encoded_img_vals.shape)
encoded_img_test = loadedEncoder.predict(x_test)
print(encoded_img_test.shape)
# using this name on classifier models


#%%
#entry block
input_shape = (50,75,64)
inputs = keras.Input(input_shape)
x = layers.Conv2D(64, (3,3), padding='same') (inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.UpSampling2D((2,2))(x)
x1 = layers.Conv2D(128, (3,3), padding='same') (x)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Activation('relu')(x1)
x1 = layers.UpSampling2D((2,2))(x1)
x2 = layers.Conv2D(256, (3,3), padding='same') (x1)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Activation('relu')(x2)
x2 = layers.UpSampling2D((2,2))(x2)
x4 = layers.Conv2D(128, (3,3), padding='same') (x2)
x4 = layers.BatchNormalization()(x4)
x4 = layers.Activation('sigmoid')(x4)

#x6 = layers.UpSampling2D((2,2))(x3)
#x7 = layers.Dropout(0.5)(x5)
#output_classifier = layers. Dense(no_units, activation='sigmoid') (x4)
classifier_model = keras.Model(inputs, x4)

classifier_model.summary()
#%%

#classifier_model.compile(optimizer='rmsprop', loss = 'binary_crossentropy')

#changing loss functions
#classifier_model.compile(optimizer='adam', loss='mean_squared_error')
#changing loss functions
#classifier_model.compile(optimizer='adam', loss = 'binary_crossentropy')

def binary_crossentropy_custom(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,K.epsilon(),1. - K.epsilon())
    loss = - tf.reduce_sum(tf.math.add( tf.math.multiply(y_true * K.log(y_pred),400), tf.math.multiply((1-y_true) * K.log(1 - y_pred),1 )  )    )
    return loss

opt = Adam(learning_rate=0.001)
classifier_model.compile(optimizer=opt, loss=binary_crossentropy_custom)



#%%
# Training method
epoch_nr = 50
train_loss_list = []
vali_loss_list = []

for epoch in range(epoch_nr):

    i = 0
    while i < len(encoded_img_xtrain):
        x_input = encoded_img_xtrain[i]
        y_input = y_train[i]
        #print(x_input.shape)
        #print(y_input.shape)
        y_input = np.reshape(y_input, (1, y_input.shape[0], y_input.shape[1], y_input.shape[2]))
        x_input = np.reshape(x_input, (1,x_input.shape[0], x_input.shape[1], x_input.shape[2]) )
        # train the model
        classifier_model.train_on_batch(x_input,y_input, return_dict=False)
        i +=1
    train_loss_list.append(classifier_model.test_on_batch(x_input,y_input))
    # change for validation data
    x_val_input = encoded_img_vals
    y_val_input = y_valid
    #print(x_val_input.shape)
    #print(y_val_input.shape)

    vali_loss_list.append(classifier_model.test_on_batch(x_val_input,y_val_input))
    print(f"epoch {epoch} completed out of {epoch_nr}")
        #train_loss
        # train_on_batch
        # train_loss_list.append()
    # use test_on_batch
    # loss_list.append()

len(train_loss_list)
len(vali_loss_list)
train_loss_list[1]
# create a line plot for train_loss_list and  vali_loss_list
# save the plot
#%%
#plotting mmodel perfo0rmance
plt.plot(train_loss_list, 'b', label = 'Tain Loss')
plt.plot(vali_loss_list, 'g', label = 'Valid Loss')
plt.title("Model Performance", fontweight='bold')
plt.legend()
plt.show()

#%%
#classifier_model.save('classifier_chicken_model.h5', overwrite = True) #saving of model weight
#loading up weight from saved model
from keras.models import load_model

loadModel = load_model('classifier_chicken_model.h5', compile=False)
loadModel.summary()

#%%
#exploting decoded image part

loadModel = load_model('autoencoder_chicken_model.h5', compile=True)
#%%
#checking out model performance
sample_test = encoded_img_test[0]
sample_test = sample_test.reshape(1, Encoder_Ouptu_Dim[0],  Encoder_Ouptu_Dim[1],  Encoder_Ouptu_Dim[2])
sample_test_y = y_test[0]
sample_test.shape
sample_test_y.shape
plt.imshow(sample_test_y[:,:,0], cmap='gray')
plt.imshow(sample_test[0,:,:,63])
pred = classifier_model.predict(sample_test)
pred
pred.shape
plt.imshow(pred[0, :, :, 0], cmap='gray')
plt.imshow(sample_test_y)
pred.shape
#%%
# test the model with test data
# create a new folder -> predictions
# save the predicted model output in that folder (as an image)

#rescalling
sample_test = encoded_img_test[0]
sample_test = sample_test.reshape(1, Encoder_Ouptu_Dim[0],  Encoder_Ouptu_Dim[1],  Encoder_Ouptu_Dim[2])
y_factor = Encoder_Input_Dim[0] / classifier_Output_Dim[0]
x_factor = Encoder_Input_Dim[1] / classifier_Output_Dim[1]
pred_rescale = cv2.resize(pred[0,:,:,:], ( Encoder_Input_Dim[1], Encoder_Input_Dim[0]))
pred_rescale.shape
plt.imshow(pred_rescale[:,:,0], cmap='gray')
plt.imshow(x_test[0])

#%%
