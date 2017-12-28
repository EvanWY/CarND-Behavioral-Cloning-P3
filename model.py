import csv
import os
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

samples = []
with open('records/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.1)

def generator(samples, batch_size=36):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size // 6):
            batch_samples = samples[offset:offset + batch_size // 6]

            images = []
            angles = []
            for batch_sample in batch_samples:
                frame_angle = float(batch_sample[3])
                
                img = cv2.imread('records/IMG/'+batch_sample[0].split('/')[-1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flip = cv2.flip(img, 1)
                images.append(img)
                angles.append(frame_angle)
                images.append(img_flip)
                angles.append(-frame_angle)
                
                img = cv2.imread('records/IMG/'+batch_sample[1].split('/')[-1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flip = cv2.flip(img, 1)
                images.append(img)
                angles.append(frame_angle + 0.15)
                images.append(img_flip)
                angles.append(-frame_angle - 0.15)
                
                img = cv2.imread('records/IMG/'+batch_sample[2].split('/')[-1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flip = cv2.flip(img, 1)
                images.append(img)
                angles.append(frame_angle - 0.15)
                images.append(img_flip)
                angles.append(-frame_angle + 0.15)

            # trim image to only see section with road
            X_train = np.array(images).reshape(-1,160,320,1)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=36)
validation_generator = generator(validation_samples, batch_size=36)

model = Sequential()
model.add(Cropping2D(cropping=((0,20), (0,0)), input_shape = (160,320,1)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(36, 5, 5, subsample = (3,3), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(96, 3, 3, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(96, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(240, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

model.summary()
print('### train sample size == {}, validation sample size == {}'.format(len(train_samples), len(validation_samples)))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(
    train_generator, 
    samples_per_epoch = len(train_samples) * 6, 
    validation_data = validation_generator, 
    nb_val_samples = len(validation_samples) * 6, 
    nb_epoch = 12)

model.save('model.h5')
exit()