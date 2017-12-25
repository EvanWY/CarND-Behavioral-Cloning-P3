import csv
import cv2
import numpy as np

images = []
measurements = []
with open('records/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        source_path = line[0]
        im_path = 'records/IMG/' + source_path.split('/')[-1]
        images.append(cv2.imread(im_path))

        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

print ()

model = Sequential()
model.add(Flatten(input_shape = X_train.shape[1:]))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')
exit()