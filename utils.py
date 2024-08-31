import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
import glob
seed=24
batch_size= 16
n_classes=3

IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3

def hwc(h,w,c):
    IMG_HEIGHT = h
    IMG_WIDTH  = w
    IMG_CHANNELS = c

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras.utils import to_categorical

train_img_dir = "/scratch/gza5dr/IrrigationTypeDetection/Experinments/Data/train_images/train/"
train_mask_dir = "/scratch/gza5dr/IrrigationTypeDetection/Experinments/Data/train_masks/train/"

img_list = sorted(os.listdir(train_img_dir))
msk_list = sorted(os.listdir(train_mask_dir))

num_images = len(os.listdir(train_img_dir))


def plot_dataset(img_num):
    img_for_plot = cv2.imread(train_img_dir+img_list[img_num], 1)
    img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)
    mask_for_plot =cv2.imread(train_mask_dir+msk_list[img_num], -1)
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(img_for_plot)
    plt.title('Image')
    plt.subplot(122)
    plt.imshow(mask_for_plot)
    plt.title('Mask')
    plt.show()
    
def preprocess_data(img, mask, num_class):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    mask = to_categorical(mask, num_class)
      
    return (img,mask)
    
def preprocess_data(img, mask, num_class):
    # img = img / 255.0  # Example normalization
    mask = tf.squeeze(mask, axis=-1)
    print(mask.shape)
    mask = tf.one_hot(tf.cast(mask, tf.int32), depth=num_class)  # Example one-hot encoding
    return img, mask

def decode_img(img, color_mode, target_size=(256, 256)):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, color_mode)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    if(color_mode == 3):
        img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the target size
    img = tf.image.resize(img, target_size)
    return img


def process_path(file_path, color_mode='rgb',img_size=(256,256)):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, color_mode=color_mode,target_size=img_size)
    return img

def trainGenerator(train_img_path, train_mask_path, batch_size, num_class, seed=None, img_size=(256,256)):

    # Create datasets for images and masks
    img_paths = tf.data.Dataset.list_files(train_img_path + '/*', seed=seed)
    mask_paths = tf.data.Dataset.list_files(train_mask_path + '/*', seed=seed)

    img_dataset = img_paths.map(lambda x: process_path(x, 3,img_size=img_size), num_parallel_calls=tf.data.AUTOTUNE)
    mask_dataset = mask_paths.map(lambda x: process_path(x, 1,img_size=img_size), num_parallel_calls=tf.data.AUTOTUNE)

    # Zip the image and mask datasets together
    dataset = tf.data.Dataset.zip((img_dataset, mask_dataset))

    # Now apply preprocess_data to each (img, mask) pair
    dataset = dataset.map(lambda img, mask: preprocess_data(img, mask, num_class), num_parallel_calls=tf.data.AUTOTUNE)

    # Continue with batching, repeating, etc.
    dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    return dataset

def plot_process_data(x):
    fig,axs = plt.subplots(3,2,figsize=(10, 10))
    for images,labels in x:
        for i in range(3):
            image = images[i]
            masks = np.argmax(labels[i],axis=-1)
            # print(labels[i])
            axs[i][0].imshow(image)
            axs[i][1].imshow(masks)
def iou(y_true, y_pred):
    # Assuming y_pred and y_true are of shape (batch_size, height, width, num_classes)
    # and contain binary values (e.g., 0 or 1).
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean(intersection / union)

def data_prep(y_true, y_pred,num_classes=3):
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=num_classes)

    # Calculate per-class TP, FP, and FN
    tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[0, 1, 2])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[0, 1, 2])
    return tp, fp, fn
    
def pre_cal(y_true, y_pred,num_classes=3):
    tp, fp, fn = data_prep(y_true, y_pred,num_classes=3)
    precision = tp / (tp + fp + 1e-5)
    return precision
    
def rec_cal(y_true, y_pred,num_classes=3):
    tp, fp, fn = data_prep(y_true, y_pred,num_classes=3)
    recall = tp / (tp + fn + 1e-5)
    return recall
    
def f1_score(y_true, y_pred,num_classes=3):
    precision =  pre_cal(y_true, y_pred,num_classes=3)
    recall = rec_cal(y_true, y_pred,num_classes=3)
    f1_scores = 2 * ((precision * recall) / (precision + recall + 1e-5)) 
    return f1_scores
    


def precision_0(y_true, y_pred,num_classes=3):
    precision =  pre_cal(y_true, y_pred,num_classes=3)
    return precision[0]
    
def precision_1(y_true, y_pred,num_classes=3):
    precision =  pre_cal(y_true, y_pred,num_classes=3)
    return precision[1]
    
def precision_2(y_true, y_pred,num_classes=3):
    precision =  pre_cal(y_true, y_pred,num_classes=3)
    return precision[2]
    
def recall_0(y_true, y_pred,num_classes=3):
    recall = rec_cal(y_true, y_pred,num_classes=3)
    return recall[0]
    
def recall_1(y_true, y_pred,num_classes=3):
    recall = rec_cal(y_true, y_pred,num_classes=3)
    return recall[1]
    
def recall_2(y_true, y_pred,num_classes=3):
    recall = rec_cal(y_true, y_pred,num_classes=3)
    return recall[2]

    
def f1_score0(y_true, y_pred,num_classes=3):
    f1_scores=f1_score(y_true, y_pred,num_classes=3)
    return f1_scores[0]

    
def f1_score1(y_true, y_pred,num_classes=3):
    f1_scores=f1_score(y_true, y_pred,num_classes=3)
    return f1_scores[1]

    
def f1_score2(y_true, y_pred,num_classes=3):
    f1_scores=f1_score(y_true, y_pred,num_classes=3)
    return f1_scores[2]



def train_model(BACKBONE,train_img_gen,val_img_gen,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,weights=None,include_top = True):
    num_train_imgs = len(os.listdir('/scratch/gza5dr/IrrigationTypeDetection/Experinments/Data/train_images/train'))
    num_val_images = len(os.listdir('/scratch/gza5dr/IrrigationTypeDetection/Experinments/Data/val_images/val'))
    steps_per_epoch = num_train_imgs//batch_size
    val_steps_per_epoch = num_val_images//batch_size
    #Use this to preprocess input for transfer learning
    model = sm.Unet(BACKBONE, encoder_weights=weights,
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    classes=n_classes, activation='softmax')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy','precision','recall',
                           f1_score,f1_score0,f1_score1,f1_score2,
                           precision_0,precision_1,precision_2,
                           recall_0,recall_1,recall_2,iou]) 
    history=model.fit(train_img_gen,
              steps_per_epoch=steps_per_epoch,
              epochs=100,
              verbose=2,
              validation_data=val_img_gen,
              validation_steps=val_steps_per_epoch)
    return model


