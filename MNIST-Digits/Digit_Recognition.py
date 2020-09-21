import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils


def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_test = np.random.normal(x_test)  # 加噪声
    x_train = x_train / 255 #normalization
    x_test = x_test / 255 #normalization

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define the network structure
    # Sequential 是线性顺序模型，多个网络层线性堆栈
    model = Sequential()

    model.add(Dense(input_dim=28*28, units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # set configuration
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # evaluate the model
    score_train = model.evaluate(x_train, y_train)
    score_test = model.evaluate(x_test,y_test)
    print('Accuracy of Training Set:', score_train[1])
    print('Accuracy of Testing Set:', score_test[1])
    print('Total loss on Testing Set:', score_test[0])

    # save the model and the weights
    model.save('Model_HDR.h5')