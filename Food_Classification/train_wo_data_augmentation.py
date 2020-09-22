import csv
import os
import cv2
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator


def readfile(path, label):
    # label 是一个boolean variable，代表不需要返回y值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)  # input image is RGB image
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))  # resize the input image as 128*128
        if label:
            y[i] = int(file.split("_")[0])
    x = x.astype('float32')
    x = x / 255
    if label:
        y = y.astype('float32')
        y = np_utils.to_categorical(y, 11)
        return x, y
    else:
        return x


# Define the information of training
batch_size = 32
epochs = 50
num_class = 11
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Food_Classification_model_1.h5'

# Read the training data, validation data ant testing data
train_x, train_y = readfile("./training", True)
print("Number of training data : ", len(train_x))
valid_x, valid_y = readfile("./validation", True)
print("Number of validation data :", len(valid_x))
test_x = readfile("./testing", False)
print("Number of testing data :", len(test_x))

# Define convolutional network structure
model = Sequential()

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(11))
model.add(Activation('softmax'))

# Set configuration
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_x, valid_y),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score train model on validation set.
scores = model.evaluate(valid_x, valid_y, verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

# Testing
result = np.argmax(model.predict(test_x), axis=-1)

# Save prediction to CSV file
with open('./saved_models/Prediction_1.csv', 'w') as f:
    f.write('ID,Category\n')
    for i, test_y in enumerate(result):
        f.write('{},{}'.format(i, test_y))
