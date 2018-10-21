#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:43:25 2018

@author: Tianyi Sun
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


images = []
labels = []

def load_data(path_name):
    for item in os.listdir(path_name):
        #print(item)
        if item.endswith('.jpg') :
                image = cv2.imread(path_name+'/'+item)
                image = cv2.resize(image, (28, 28))
                    
                images.append(image)                
                labels.append(path_name)  
    return images, labels                              

path_rose ='./images/rose'
path_sunflower ='./images/sunflower'

images1,labels1 = load_data(path_rose)   
images2,labels2 = load_data(path_sunflower)

images = images1 + images2
labels = labels1 + labels2
  
images = np.array(images)

i=0
labelsTemp=[]
for label in labels:
    if label.endswith('rose'):
        labelsTemp.append(0)
        i+=1
    else:
        labelsTemp.append(1)
        i+=1
        
labels = labelsTemp

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.30, random_state=42)


class_names = ['rose', 'sunflower']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


predictions = model.predict(test_images)


#show the result
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
 

  
#show the top 100 pictures
num_rows = 100
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)


