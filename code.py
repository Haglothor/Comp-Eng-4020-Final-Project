from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
%matplotlib inline

main_directory = '30by30 images' #Add personal directory here
transfer_directory = 'Transfer Dataset' #Add personal directory here

#Pull and split datasets
component_train = tf.keras.preprocessing.image_dataset_from_directory(main_directory,
                                                                      validation_split = 0.2,
                                                                      subset = 'training',
                                                                      seed = 123,
                                                                      batch_size = 16,
                                                                      color_mode = 'grayscale',
                                                                      image_size = (30,30)
                                                                     )
component_test = tf.keras.preprocessing.image_dataset_from_directory(main_directory,
                                                                      validation_split = 0.2,
                                                                      subset = 'validation',
                                                                      seed = 123,
                                                                      batch_size = 16,
                                                                      color_mode = 'grayscale',
                                                                      image_size = (30,30)
                                                                     )
transfer_train = tf.keras.preprocessing.image_dataset_from_directory(transfer_directory,
                                                                      validation_split = 0.2,
                                                                      subset = 'training',
                                                                      seed = 123,
                                                                      batch_size = 16,
                                                                      color_mode = 'grayscale',
                                                                      image_size = (30,30)
                                                                     )
transfer_test = tf.keras.preprocessing.image_dataset_from_directory(transfer_directory,
                                                                      validation_split = 0.2,
                                                                      subset = 'validation',
                                                                      seed = 123,
                                                                      batch_size = 16,
                                                                      color_mode = 'grayscale',
                                                                      image_size = (30,30)
                                                                     )
#Tune Training for Optimization
AUTOTUNE = tf.data.AUTOTUNE
component_train = component_train.cache().prefetch(buffer_size=AUTOTUNE)
component_test = component_test.cache().prefetch(buffer_size=AUTOTUNE)
transfer_train = component_train.cache().prefetch(buffer_size=AUTOTUNE)
transfer_test = component_test.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 3

feature_layers_1 = [
  Conv2D(32, 3, activation='relu', input_shape = (30,30,1)),
  MaxPooling2D(),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Flatten()
  ]

classification_layers_1 = [
  Dense(128),
  Activation('relu'),
  Dropout(0.25),
  Dense(num_classes),
  Activation('softmax')
]

model_1 = Sequential(feature_layers_1 + classification_layers_1)

model_1.summary()

model_1.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model_1.fit(
  component_train,
  validation_data=component_test,
  epochs=3
)

for layer in feature_layers_1:
    layer.trainable = False

model_1 = Sequential(feature_layers_1 + classification_layers_1)

model_1.summary()

model_1.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model_1.fit(
  transfer_train,
  validation_data=transfer_test,
  epochs=3
)

feature_layers_2 = [
  Conv2D(32, 3, activation='relu', input_shape = (30,30,1)),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, activation='relu'),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Flatten(),
  ]

classification_layers_2 = [
  Dense(128, activation='relu'),
  Dense(128, activation='relu'),
  Dropout(0.25),
  Dense(num_classes),
  Activation('softmax')
]

model_2 = Sequential(feature_layers_2 + classification_layers_2)

model_2.summary()

model_2.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model_2.fit(
  component_train,
  validation_data=component_test,
  epochs=3
)

for layer in feature_layers_2:
    layer.trainable = False

model_2 = Sequential(feature_layers_2 + classification_layers_2)

model_2.summary()

model_2.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model_2.fit(
  transfer_train,
  validation_data=transfer_test,
  epochs=3
)

feature_layers_3 = [
  Conv2D(32, 3, activation='relu', input_shape = (30,30,1)),
  Conv2D(32, 3, activation='relu'),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, activation='relu'),
  Conv2D(32, 3, activation='relu'),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Flatten()
  ]

classification_layers_3 = [
  Dense(512, activation='relu'),
  Dense(256, activation='relu'),
  Dense(128, activation='relu'),
  Dropout(0.25),
  Dense(num_classes),
  Activation('softmax')
]

model_3 = Sequential(feature_layers_3 + classification_layers_3)

model_3.summary()

model_3.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model_3.fit(
  component_train,
  validation_data=component_test,
  epochs=3
)

for layer in feature_layers_3:
    layer.trainable = False

model_3 = Sequential(feature_layers_3 + classification_layers_3)

model_3.summary()

model_3.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model_3.fit(
  transfer_train,
  validation_data=transfer_test,
  epochs=3
)
