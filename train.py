import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

data_path = "/content/traffic-sign-dataset-classification.zip"

with ZipFile(data_path, "r") as zip_ref:
    zip_ref.extractall("/content")

print("Dataset extracted")

dataset = "/content/traffic_Data/DATA"

labelfile = pd.read_csv("/content/labels.csv")

img = cv2.imread("/content/traffic_Data/DATA/10/010_0011.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()


img = cv2.imread("/content/traffic_Data/DATA/23/023_0001.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()

labelfile.head()
labelfile.tail()


import tensorflow as tf

dataset = "/content/DATA"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_numbers = train_ds.class_names

class_names = []
for i in class_numbers:
    class_names.append(labelfile['Name'][int(i)])


plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomFlip("horizontal_and_vertical")
])


model = Sequential()
model.add(data_augmentation)
model.add(Rescaling(1./255))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))

model.add(Dense(len(labelfile), activation='softmax'))