# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:17:53 2023

@author: anadjj
"""

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from helper import create_dir
from matplotlib import pyplot as plt
from data import load_aug_data, tf_dataset
from model import build_model
from visualization import visualization


files_dir = os.path.join("files", "aug")
model_file = os.path.join(files_dir, "unet-aug.h5")
log_file = os.path.join(files_dir, "log-aug.csv")

create_dir(files_dir)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

if __name__ == "__main__":

    #include of np.random gave worst results
    #np.random.seed(42)
    #tf.random.set_seed(42)
    ## Dataset
    #path = "PNG/"
    #Path
    dataset_path = os.path.join("dataset", "aug/")
    
    path = "PNG/"
    (train_x, train_y), (valid_x, valid_y) = load_aug_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 2
    #epochs = 20

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)
    '''
    for x, y in train_dataset:
      print(x.shape, y.shape)
    '''
    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_file),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger(log_file),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    history = model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)

    visualization(history)