import csv
import cv2
import numpy as np
import os
import math
from random import shuffle
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

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

IsUsingGenerator = False
debugOn = False


samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

recover_samples = []
with open('data_recover/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		recover_samples.append(line)

sharpturn_samples = []
with open('data_sharp_turn/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		sharpturn_samples.append(line)


angles = []
images = []
for sample in samples:
    name = 'data/IMG/'+ sample[0].split('/')[-1]
    center_image = cv2.imread(name)
    center_angle = float(sample[3])
    images.append(center_image)
    angles.append(center_angle)
    images.append(cv2.flip(center_image, 1))
    angles.append(-center_angle)

# for sample in recover_samples:
#     name = 'data_recover/IMG/'+ sample[0].split('/')[-1]
#     center_image = cv2.imread(name)
#     center_angle = float(sample[3])
#     images.append(center_image)
#     angles.append(center_angle)


if debugOn:
    print("size of angles:", len(angles))
    print("angles[14]", angles[14])


X_train = np.array(images)
y_train = np.array(angles)



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D

# Create the Sequential model
row, col, ch = 66, 320, 3  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 34), (0, 0))))
model.add(Conv2D(24, (5, 5), padding="valid",subsample=(2, 2), activation="relu")) # input shape=(66, 320, 3)
model.add(Conv2D(36, (5, 5), padding="valid",subsample=(2, 2), activation="relu")) # input shape=(31, 158, 24)
model.add(Conv2D(48, (5, 5), padding="valid",subsample=(2, 2), activation="relu")) # input shape=(14, 77, 36)
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu")) # input shape=(5, 37, 48)
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu")) # input shape=(3, 35, 64)
model.add(Flatten())                                              # input shape=(1, 33, 64)_
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('model created')
from keras.utils import plot_model
plot_model(model, to_file='model.png')

# compile and fit the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

# history_object = model.fit_generator(train_generator,
#             steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
#             validation_data=validation_generator, 
#             validation_steps=math.ceil(len(validation_samples)/batch_size), 
#             epochs=1, verbose=1)

model.save('model.h5')
print('model saved')


### plot the training and validation loss for each epoch
# import matplotlib.pyplot as plt
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()