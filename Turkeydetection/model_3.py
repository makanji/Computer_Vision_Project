# library importations
import keras.backend as keras  # this doesnt suppoort model input command
# import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# data processing
# %%
# parameters definitions
Encoder_Input_Dim = (800, 1200, 1)
Encoder_Ouptu_Dim = (50, 75, 64)
color_channel = 1
classifier_Output_Dim = (400, 600, 4)
input_shape = (50, 75, 64)

#%%
# dataset importation using 300 images
x_array = np.load('/Home/makanji/Master_Thesis/Datasets/Turkey_arr/x_array.npy')
y_array = np.load('/Home/makanji/Master_Thesis/Datasets/Turkey_arr/y_array.npy')
print(x_array.shape)
print(y_array.shape)
#%%
# splitting into respective groups
# for x_arrays
from sklearn.utils import shuffle

x_array_1, y_array_1 = shuffle(x_array, y_array)
id = np.arange(len(x_array_1))
id_train = int(len(id) * 0.8)  # Train 80%
id_test = int(len(id) * (0.8 + 0.05))  # Valid 15%, Test 5%
train_set, test_set, val_set = np.split(id, (id_train, id_test))
# Indexing dataset subsets
x_train = x_array_1[train_set, :, :]  # Train set
x_valid = x_array_1[val_set, :, :]  # Valid set
x_test = x_array_1[test_set, :, :]
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
y_train = y_array_1[train_set, :, :, :]  # Train set
y_valid = y_array_1[val_set, :, :, :]  # Valid set
y_test = y_array_1[test_set, :, :, :]
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

#%%

# reshaping foe model channel save
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], color_channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], color_channel)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], color_channel)
# %%
# load the encoder
from tensorflow.keras.models import load_model

loadedEncoder = load_model("Turkey_Encoder.h5", compile=False)

# %%
# creating coded image version
encoded_img_xtrain = loadedEncoder.predict(x_train)
print(encoded_img_xtrain.shape)
encoded_img_vals = loadedEncoder.predict(x_valid)
print(encoded_img_vals.shape)
encoded_img_test = loadedEncoder.predict(x_test)
print(encoded_img_test.shape)

# model building
#%%
# entry block
input_shape = (50, 75, 64)
# inputs = keras.Input(input_shape)
inputs = tf.keras.Input(input_shape)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(128, (3, 3), padding='same',  activation='relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(256, (3, 3), padding='same',  activation='relu', kernel_regularizer =tf.keras.regularizers.l2( l=0.01))(x)
x = layers.BatchNormalization()(x)

#x = layers.Activation('relu')(x)
x = layers.UpSampling2D((2, 2))(x)

# second block
x1 = layers.Conv2D(256, (3, 3), padding='same' , activation='relu')(x)
x1 = layers.BatchNormalization()(x1)

x1 = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer =tf.keras.regularizers.l2( l=0.01),  activation='relu')(x1)
x1 = layers.BatchNormalization()(x1)

# x1 = layers.Conv2D(1024, (3,3), padding='same') (x1)
# x1 = layers.BatchNormalization()(x1)

#x1 = layers.Activation('relu')(x1)
x1 = layers.UpSampling2D((2, 2))(x1)
x1 = layers.Dropout(0.2)(x1)

# 3rd block

# second block
# x2 = layers.Conv2D(512, (3,3), padding='same') (x1)
# x2 = layers.BatchNormalization()(x2)

x2 = layers.Conv2D(256, (3, 3), padding='same',   activation='relu')(x1)
x2 = layers.BatchNormalization()(x2)

x2 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x2)
x2 = layers.BatchNormalization()(x2)

#x2 = layers.Activation('relu')(x2)
#x2 = layers.Dropout(0.2)(x2)
x2 = layers.UpSampling2D((2, 2))(x2)

# last block
x3 = layers.Conv2D(32, (3, 3), padding='same',  activation='relu')(x2)
x3 = layers.BatchNormalization()(x3)
#x3 = layers.Activation('sigmoid')(x3)
x3 = layers.Dropout(0.5)(x3)
# x3 = layers.Conv2D(32, (3,3), padding='same') (x3)
# x3 = layers.BatchNormalization()(x3)
# x3 = layers.Activation('sigmoid')(x3)

x3 = layers.Conv2D(4, (3, 3), padding='same')(x3)
x3 = layers.BatchNormalization()(x3)
x3 = layers.Activation('sigmoid')(x3)

# classifer
# classifier_model = keras.Model(inputs, x3)
classifier_model = tf.keras.Model(inputs, x3)
# classifier_model = keras.Model(inputs, x3)
classifier_model.summary()

#%%
# compiler

# classifier_model.compile(optimizer='rmsprop', loss = 'binary_crossentropy')

# changing loss functions
# classifier_model.compile(optimizer='adam', loss='mean_squared_error')
# changing loss functions
classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

'''
def binary_crossentropy_custom(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,keras.epsilon(),1. - keras.epsilon())
    loss = - tf.reduce_sum(tf.math.add( tf.math.multiply(y_true * keras.log(y_pred),400), tf.math.multiply((1-y_true) * keras.log(1 - y_pred),1)))
    return loss

#opt = Adam(learning_rate=0.001)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
classifier_model.compile(optimizer=opt, loss=binary_crossentropy_custom)'''

#%%
# Training method
epoch_nr = 50
train_loss_list = []
vali_loss_list = []
vali_hist = []

for epoch in range(epoch_nr):

    i = 0
    while i < len(encoded_img_xtrain):
        x_input = encoded_img_xtrain[i]
        y_input = y_train[i]
        # print(x_input.shape)
        # print(y_input.shape)
        y_input = np.reshape(y_input, (1, y_input.shape[0], y_input.shape[1], y_input.shape[2]))
        x_input = np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1], x_input.shape[2]))
        # train the model
        classifier_model.train_on_batch(x_input, y_input, return_dict=False)
        i += 1
    #print('one completed')
    train_loss_list.append(classifier_model.test_on_batch(x_input, y_input))
    # change for validation data
    x_val_input = encoded_img_vals
    y_val_input = y_valid
    # print(x_val_input.shape)
    # print(y_val_input.shape)

    vali_loss_list.append(classifier_model.test_on_batch(x_val_input, y_val_input))
    print(f"epoch {epoch} completed out of {epoch_nr}")

'''
    #saving up the validations sets
    x_val_encoded = loadedEncoder.predict(x_valid)
    x_val_encoded_final =  np.zeros(shape=(x_val_encoded.shape[0], x_val_encoded.shape[1], x_val_encoded.shape[2], x_val_encoded.shape[3]))
    print('one completed')
    for sample_img in range(x_valid.shape[0]):
        for color_chan in range(x_valid.shape[3]):
            orig_chan_resized = cv2.resize(x_valid[sample_img, :, :, color_chan].shape, (input_shape[1], input_shape[0]))
            print('this completed')
            x_val_encoded_final[sample_img, : , : , color_chan] =  orig_chan_resized
            print('this completed')
        x_val_encoded_final[sample_img, : , : , (color_chan + 1): (x_val_encoded_final.shape[3] + 1)] = x_val_encoded.shape [sample_img, :, :, :]

        print('one completed two')
    x_val_encoded_final = x_val_encoded_final / x_val_encoded_final.max()
    print('one completed finally')

    vali_hist.apend( classifier_model.test_on_batch(x_val_encoded_final[0:1, :,:,:], y_val_input) / x_val_encoded_final.shape[0])

    tf.keras.Model.save(classifier_model,"/Home/makanji/Master_Thesis/turckey_classifier" + "_Epoch_" + str(epoch) + ".h5")

    #print("\n Epoch: %s \n Training Loss: %s \n Validation loss: %s" % (str(epoch), str(train_loss_list)))


    print(f"epoch {epoch} completed out of {epoch_nr}")'''

# train_loss
# train_on_batch
# train_loss_list.append()
# use test_on_batch
# loss_list.append()

#%%
orig_chan_resized.shape
plt.imshow(orig_chan_resized)
len(x_val_encoded_final)
x_val_encoded_final.shape
plt.imshow(x_val_encoded_final[0, :, :, 4])

#%%
# plotting mmodel perfo0rmance
plt.plot(train_loss_list, 'b', label='Tain Loss')
plt.plot(vali_loss_list, 'g', label='Valid Loss')
plt.title("Model Performance", fontweight='bold')
plt.legend()
plt.show()

#%%
# classifier_model.save('classifier_Turkey_model_v1.h5',  overwrite = True) #saving of model weight
# loading up weight from saved model
from keras.models import load_model

loadModel = load_model('classifier_Turkey_model.h5', compile=False)
loadModel.summary()

#%%
# checking out model performance
sample_test = encoded_img_test[0]
sample_test = sample_test.reshape(1, Encoder_Ouptu_Dim[0], Encoder_Ouptu_Dim[1], Encoder_Ouptu_Dim[2])
sample_test.shape
plt.imshow(sample_test[0, :, :, 63])
pred = classifier_model.predict(sample_test)
pred
pred.shape
plt.imshow(pred[0, :, :, 1], cmap='gray')
#%%

# rescalling
sample_test = encoded_img_test[0]
sample_test = sample_test.reshape(1, Encoder_Ouptu_Dim[0], Encoder_Ouptu_Dim[1], Encoder_Ouptu_Dim[2])
y_factor = Encoder_Input_Dim[0] / classifier_Output_Dim[0]
x_factor = Encoder_Input_Dim[1] / classifier_Output_Dim[1]
pred_rescale = cv2.resize(pred[0, :, :, :], (Encoder_Input_Dim[1], Encoder_Input_Dim[0]))
pred_rescale.shape
plt.imshow(pred_rescale[:, :, 0], cmap='gray')
plt.imshow(x_test[0])
