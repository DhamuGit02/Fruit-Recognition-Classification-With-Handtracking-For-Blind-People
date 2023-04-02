from keras.models import load_model
import cv2
import os
from Classifire_model import test_generator

model = load_model('Fruit_Classifier_model_CNN.h5')
img = cv2.imread(r'D:\SEMESTER V MINI PROJECT\fruits-360_dataset\fruits-360\Training\Apple Golden 1\0_100.jpg')
fruit_name = [folder for folder in os.listdir(r'D:/SEMESTER V MINI PROJECT/fruits-360_dataset/fruits-360/Training')]
print(model.predict(cv2.imread(r'D:\SEMESTER V MINI PROJECT\fruits-360_dataset\fruits-360\Training\Cherry 1\0_100.jpg')))
