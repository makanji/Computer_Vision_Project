#%%
#libr   maries importations
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

#%%
# define the main parameters
Encoder_Input_Dim = (800,1200,3)
Encoder_Ouptu_Dim = (75,50,64)

classifier_Output_Dim = (400,600,4)

#%%
# load the encoder
loadedEncoder = tf.keras.models.load_model("Encoder.h5", compile =False)


#%%
#folder descriptions
home_folder = "/Home/makanji/Master_Thesis/Datasets/Turkey1_Grp/"
turkey_grp1 = "/Home/makanji/Master_Thesis/Datasets/Turkey1_Grp/Turkey_Grp1/"
turkey_grp2 = "/Home/makanji/Master_Thesis/Datasets/Turkey1_Grp/Turkey_Grp2/"
turkey_grp3 = "/Home/makanji/Master_Thesis/Datasets/Turkey1_Grp/Turkey_Grp3/"

os.listdir(home_folder)
csv_file = home_folder + 'Turkey_Grp1Annotation.csv'
csv_file_2 = home_folder + 'Turkey_Grp2Annotation.csv'
csv_file_3 = home_folder + 'Turkey_Grp3Annotation.csv'
#%%
#folder importation for the images
raw_images = []
folder_names = os.listdir(turkey_grp1)
len(folder_names)
for img_index in range(len(folder_names)):
    print(img_index)
    read_img = cv2.imread(turkey_grp1 + folder_names[img_index], 0)
    print(folder_names[img_index] + f"image was suffecesully imported with dimension " + str( read_img.shape))
    raw_images.append(read_img)
print(len(raw_images))
#%%
#csv folder importations
#file descriptions
annotation_1 = pd.read_csv(csv_file)
annotation_1.head(5)
annotation_1.tail(5)
annotation_1.shape



#%%
#for all three body parts
#getting uniques names from files
unique_names = pd.unique((annotation_1.iloc[:, 1]))
print(len(unique_names))
annotated_img_head = []
annotated_img_body = []
annotated_img_tail = []
binary_images = []
orig_images = []

current_binary.shape
plt.imshow(binary_images[0])

for img_nr in range(len(unique_names)):
    print(img_nr)
    #extract the name from the annonation folder
    current_name = unique_names[img_nr]
    #read the main image with the annotation coordinates
    orig_image = cv2.imread(turkey_grp1 + current_name, 0)
    orig_dimension = orig_image.shape
    y_factor = orig_dimension[0] / Encoder_Input_Dim[0]
    x_factor = orig_dimension[1] / Encoder_Input_Dim[1]
    orig_image = cv2.resize(orig_image, (Encoder_Input_Dim[1], Encoder_Input_Dim[0])) # for cv2: index x-dim first, then y_dim
    current_img = copy.deepcopy(orig_image)
    # create the binary image
    current_binary = np.zeros(shape=(current_img.shape[0], current_img.shape[1],3),dtype = float)
    current_bin_head = copy.deepcopy(current_binary[:, :, 0])
    current_bin_body = copy.deepcopy(current_binary[:, :, 1])
    current_bin_tail = copy.deepcopy(current_binary[:, :, 2])
    #scalling the whole frame
    current_img = current_img/255
    print('Img ' + current_name + ' index ' + str( img_nr) + ' shape ' + str(current_img.shape) + ' imported')
    df_unique_id = annotation_1[annotation_1['fileName'].str.contains(current_name)]
    #for head part
    for entry_hd in range(len(df_unique_id)):
        #indexing through the column
        coord = df_unique_id.iloc[entry_hd,2]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(int(coord_x) / x_factor)
        coord_y = coord.split(',')[1]
        coord_y = int(int(coord_y) / y_factor)
        current_img_cir_hd = cv2.circle(current_img, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
        #current_bin_head[(coord_y-8) : (coord_y+8),( coord_x-8) :(coord_x+8)] = 1
        current_bin_head = cv2.circle(current_bin_head, (coord_x, coord_y), radius=100, color=1.00, thickness=cv2.FILLED)
    annotated_img_head.append(current_img_cir_hd)
    current_img = cv2.imread(turkey_grp1 + current_name, 0)
    # scalling the whole frame
    current_img = current_img / 255
    #extraction for body parts iterations
    for entry_bd in range(len(df_unique_id)):
        #indexing through the column
        coord = df_unique_id.iloc[entry_bd,3]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(coord_x)
        coord_y = coord.split(',')[1]
        coord_y = int(coord_y)
        current_img_cir_bd = cv2.circle(current_img, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
        current_bin_body[(coord_y - 30): (coord_y + 30), (coord_x - 30):(coord_x + 30)] = 1
    annotated_img_body.append(current_img_cir_bd)
    current_img = cv2.imread(turkey_grp1 + current_name, 0)
    # scalling the whole frame
    current_img = current_img / 255
    #extraction for tail part interations
    for entry_tl in range(len(df_unique_id)):
        #indexing through the column
        coord = df_unique_id.iloc[entry_tl,4]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(coord_x)
        coord_y = coord.split(',')[1]
        coord_y = int(coord_y)
        current_img_cir_tl = cv2.circle(current_img, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
        current_bin_tail[(coord_y - 30): (coord_y + 30), (coord_x - 30):(coord_x + 30)] = 1
    annotated_img_tail.append(current_img_cir_tl)
    current_binary[:,:,0] = current_bin_head
    current_binary[:, :, 1] = current_bin_body
    current_binary[:, :, 2] = current_bin_tail
    binary_images.append(current_binary)
    orig_images.append(orig_image)

#%%
# transform the lists into array
x_array = np.array(orig_images)
y_array = np.array(binary_images)

#model_name.train_on_batch(x_array, y_array)

#%%
#exlorations of the above functions
print(len(annotated_img_head))
print(len(annotated_img_body))
print(len(annotated_img_tail))
#checking images explore out
sample_extract = annotated_img_head[4]
plt.imshow(sample_extract, cmap='gray')
plt.imshow(annotated_img_body[4], cmap = 'gray')
plt.imshow(annotated_img_tail[4], cmap = 'gray')


#%%
#extraction for binaRY CHANNELS
bnry_head = []
bnry_body = []
bnry_tail = []
bnry_hd_tl= []
binary_image = np.ndarray((2160, 3840), dtype=float)
#using uniques image folder names
print(len(unique_names))
for img_bnry_index in range(len(unique_names)):
    current_name = unique_names[img_bnry_index]
    df_unique_id = annotation_1[annotation_1['fileName'].str.contains(current_name)]
    binary_image = np.ndarray((2160, 3840), dtype=float)
    #extraction of coord parameter using name
    for bnry_indx_head in range(len(df_unique_id)):
        coord = df_unique_id.iloc[bnry_indx_head, 2]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(coord_x)
        coord_y = coord.split(',')[1]
        coord_y = int(coord_y)
        current_img_cir_hd = cv2.circle(binary_image, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
    bnry_head.append(current_img_cir_hd)
    binary_image = np.ndarray((2160, 3840), dtype=float)
    # extraction for body parts iterations
    for entry_bd in range(len(df_unique_id)):
        # indexing through the column
        coord = df_unique_id.iloc[entry_bd, 3]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(coord_x)
        coord_y = coord.split(',')[1]
        coord_y = int(coord_y)
        current_img_cir_bd = cv2.circle(binary_image, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
    bnry_body.append(current_img_cir_bd)
    binary_image = np.ndarray((2160, 3840), dtype=float)
    # extraction for tail part interations
    for entry_tl in range(len(df_unique_id)):
        # indexing through the column
        coord = df_unique_id.iloc[entry_tl, 4]
        coord = str(coord)
        coord_x = coord.split(',')[0]
        coord_x = int(coord_x)
        coord_y = coord.split(',')[1]
        coord_y = int(coord_y)
        current_img_cir_tl = cv2.circle(binary_image, (coord_x, coord_y), radius=10, color=1.00, thickness=cv2.FILLED)
    bnry_tail.append(current_img_cir_tl)
    binary_image = np.ndarray((2160, 3840), dtype=float)
    #for binary head and tail channel
    for bnry_indx_head, bnry_indx_tail in enumerate(range(len(df_unique_id))):
        coord_hd = df_unique_id.iloc[bnry_indx_head, 2]
        coord_hd = str(coord_hd)
        coord_x_hd = coord_hd.split(',')[0]
        coord_x_hd = int(coord_x_hd)
        coord_y_hd = coord_hd.split(',')[1]
        coord_y_hd = int(coord_y_hd)
        coord_tl = df_unique_id.iloc[bnry_indx_tail, 4]
        coord_tl = str(coord_tl)
        coord_x_tl = coord_tl.split(',')[0]
        coord_x_tl = int(coord_x_tl)
        coord_y_tl = coord_tl.split(',')[1]
        coord_y_tl = int(coord_y_tl)
        current_img_ln = cv2.line(binary_image, (coord_x_hd, coord_y_hd), (coord_x_tl, coord_y_tl), color=1.00,
                                  thickness=10)
    bnry_hd_tl.append(current_img_ln)
    print('img index ' + str(img_bnry_index ) + ' named ' + current_name + ' is successfully processed')

#%%
#result explorations
print(len(bnry_hd_tl))
print(len(bnry_head))
print(len(bnry_body))
print(len(bnry_tail))
sample_extract = bnry_hd_tl[0]
plt.imshow(sample_extract, cmap='gray')
sample_extract = bnry_head[img_nr]
plt.imshow(sample_extract, cmap='gray')
plt.imshow(bnry_head[1],cmap= 'gray')
plt.imshow(bnry_body[1],cmap= 'gray')
plt.imshow(bnry_tail[99],cmap= 'gray')
plt.imshow(bnry_hd_tl[99],cmap= 'gray')


#%%
#joining all images together to form one channel
bnry_head_array = np.asarray(bnry_head)
bnry_body_array = np.asarray(bnry_body)
bnry_tail_array = np.asarray(bnry_tail)
bnry_hd_tl_array = np.asarray(bnry_hd_tl)
print(bnry_hd_tl_array.shape)
all_channels_bnry = cv2.merge([bnry_head_array, bnry_body_array, bnry_tail_array, bnry_hd_tl_array])
plt.imshow(bnry_hd_tl_array[0,:,:])
print(all_channels_bnry.shape)
plt.imshow(all_channels_bnry[0,:,:,:], cmap = 'gray')
#%%
