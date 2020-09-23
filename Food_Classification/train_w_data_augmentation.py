
# Validation loss: 1.3684558868408203
# Validation accuracy: 0.6889212727546692

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
    x = x/255
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
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'Keras_Food_Classification_model_2.h5'

# Read the training data, validation data ant testing data
train_x, train_y = readfile("./training", True)
print("Number of training data : ", len(train_x))
valid_x, valid_y = readfile("./validation",True)
print("Number of validation data :", len(valid_x))
test_x = readfile("./testing", False)
print("Number of testing data :", len(test_x))

# Define convolutional network structure
model = Sequential()

model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), strides=(1,1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), strides=(1,1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3,3), strides=(1,1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3,3), strides=(1,1), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

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
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(valid_x, valid_y),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format='channels_last',
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.2)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_x)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_x, train_y,
                                     batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(valid_x, valid_y),
                                     workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score train model on validation set.
scores = model.evaluate(valid_x,valid_y , verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

# Testing
result = np.argmax(model.predict(test_x), axis=-1)

# Save prediction to CSV file
with open('./saved_models/Prediction_2.csv', 'w') as f:
    f.write('ID,Category\n')
    for i, test_y in enumerate(result):
        f.write('{},{}'.format(i, test_y))