import os

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

training_path = r'D:\SEMESTER V MINI PROJECT\fruits-360_dataset\fruits-360\Training\\'
testing_path = r'D:\SEMESTER V MINI PROJECT\fruits-360_dataset\fruits-360\Test\\'

model = Sequential()
# Convolution 2D is a mathematical function to generate convolution kernel that binds with input layers to produce tensor outputs
# first param is number of filters (a filter is a matrix or kernel of mxm of odd order which is convoluted with input images to reduce spatial volume of output image
# second param is kernel size integer/ tuple (3, 3) or only 3
model.add(Conv2D(128, 3, activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(131, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   zoom_range = 0.3
                                   )
test_datagen  = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(training_path,
                                                    target_size=(100, 100),
                                                    batch_size=32,
                                                    color_mode="rgb",
                                                    class_mode="categorical")
test_generator = test_datagen.flow_from_directory(testing_path,
                                                  target_size=(100, 100),
                                                  batch_size=32,
                                                  color_mode="rgb",
                                                  class_mode="categorical")

hist = model.fit_generator(generator = train_generator,
                           steps_per_epoch=50,
                           epochs=50,
                           validation_data=test_generator,
                           validation_steps=50)

model.save('Fruit_Classifier_model_CNN.h5')