from operator import mod

from tensorflow.python.keras.layers.core import Dropout
from cnn_utils import *

from tensorflow.keras.datasets.mnist import load_data
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.losses import sparse_categorical_crossentropy

def train_model(x_train, y_train, x_test, y_test):
    # y_train = convert_to_one_hot(y_train, 6).T
    # y_test = convert_to_one_hot(y_test, 6).T
    in_shape = x_train.shape[1:]
    # determine the number of classes
    n_classes = len(unique(y_train))
    print(in_shape, n_classes)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0



    model = Sequential()
    model.add(Conv2D(16, (4,4), padding= 'same', activation='relu', input_shape = in_shape))
    model.add(MaxPool2D(pool_size=(8,8), strides=(8,8), padding='same'))
    model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(4,4), strides=(4,4), padding='same'))
    # model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(4,4), strides=(4,4), padding='same'))
    model.add(Flatten())    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(600, activation='relu'))
    model.add(Dense(700, activation='relu'))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # define loss and optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0)
    # evaluate the model
    loss, acc = model.evaluate(x_train, y_train, verbose=0)
    print('Accuracy on test: %.3f' % acc)
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy on test: %.3f' % acc)
    # make a prediction
    image = x_train[0]
    yhat = model.predict(asarray([image]))
    print('Predicted: class=%d' % argmax(yhat))

    val = input("Save(y/n): ")
    if(val == 'y'):
        model.save('model/my_model.h5')
    else:
        print("Model not save.")


def load_my_model():
    import tensorflow.keras.models as md
    return md.load_model('model/my_model.h5')


def m_predict():
    from PIL import Image
    fname = "images/3.jpg"
    image = Image.open(fname)
    image = image.resize((64, 64))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    plt.imshow(image)
    plt.show()
    model = load_my_model()
    yhat = model.predict(asarray([image]))
    print('Predicted: class=%d' % argmax(yhat))


m_predict()


# X_train, Y_train, X_test, Y_test, classes = load_dataset()
# print(X_train.shape)
# print(Y_train.shape)
# train_model(X_train, Y_train.T, X_test, Y_test.T)


# (x_train, y_train), (x_test, y_test) = load_data()
# print(x_train.shape)
# print(y_train.shape)