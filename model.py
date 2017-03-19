import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#Read driving log which includes angles as measurements
lines = []
with open(‘./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
     lines.append(line)

images = []
measurements = []

#Gather all the images and angle data into two lists
#Corrections are applied to the angles
for line in lines:
  for i in range(3):
    source_path = line[i]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = “./data/IMG/" + filename
    image = cv2.imread(local_path)
    images.append(image)
  correction = 0.2
  measurement = float(line[3])
  measurements.append(measurement)
  measurements.append(measurement+correction)
  measurements.append(measurement-correction)

#Augment the image files from the cameras by flipping them
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  flipped_image = cv2.flip(image, 1)
  flipped_measurement = float(measurement) * -1.0
  augmented_images.append(flipped_image)
  augmented_measurements.append(flipped_measurement)

#Denote the images and angle measurements as arrays
x_train = np.array(images)
y_train = np.array(measurements)

#Build a model
#Architecture is described in the markdown file
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) 
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Run the model
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True)

#Save the model
model.save('model.h5')