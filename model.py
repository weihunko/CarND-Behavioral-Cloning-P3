import csv
import cv2
import numpy as np
import os
import math
from random import shuffle

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for line in reader:
		if i == 0:
			i = i + 1
			continue
		samples.append(line)


shuffle(samples)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.35)


import sklearn


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)




from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create the Sequential model
ch, row, col = 3, 90, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Conv2D(24, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Conv2D(36, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Conv2D(48, (5, 5), padding="valid", activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('model created')


# compile and fit the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=1, verbose=1)


model.save('model.h5')
print('model saved')


# ### plot the training and validation loss for each epoch
# import matplotlib.pyplot as plt
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()