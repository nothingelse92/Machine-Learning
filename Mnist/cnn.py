import gzip, cPickle
import numpy as np
import os
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]
def CNN(activation):
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("mnist.pkl.gz")
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, 28, 28)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)
    XCV = Xtest.reshape(XCV.shape[0], 1, 28, 28)
    # 0~9 ten classes
    ytrain = np_utils.to_categorical(ytrain, 10)
    ytest = np_utils.to_categorical(ytest, 10)
    yCV = np_utils.to_categorical(yCV, 10)
    # Build the model
    model = Sequential()
    model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(1,28,28)))
    model.add(Activation(activation))
    model.add(Convolution2D(32,3,3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
	# fit module
    print "fit module"
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit(Xtrain,ytrain,batch_size=100,nb_epoch=20,verbose=1,validation_data=(XCV,yCV))
    score = model.evaluate(Xtest,ytest, verbose=0)
    print score[0]
    print score[1]
def CNN_NO_DropOut(activation):
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("mnist.pkl.gz")
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, 28, 28)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)
    XCV = Xtest.reshape(XCV.shape[0], 1, 28, 28)
    # 0~9 ten classes
    ytrain = np_utils.to_categorical(ytrain, 10)
    ytest = np_utils.to_categorical(ytest, 10)
    yCV = np_utils.to_categorical(yCV, 10)
    # Build the model
    model = Sequential()
    model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(1,28,28)))
    model.add(Activation(activation))
    model.add(Convolution2D(32,3,3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dense(10))
    model.add(Activation('softmax'))
	# fit module
    print "fit module"
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit(Xtrain,ytrain,batch_size=100,nb_epoch=20,verbose=1,validation_data=(XCV,yCV))
    score = model.evaluate(Xtest,ytest, verbose=0)
    print score[0]
    print score[1]
def CNN_3_layer(activation):
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("mnist.pkl.gz")
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, 28, 28)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)
    XCV = Xtest.reshape(XCV.shape[0], 1, 28, 28)
    # 0~9 ten classes
    ytrain = np_utils.to_categorical(ytrain, 10)
    ytest = np_utils.to_categorical(ytest, 10)
    yCV = np_utils.to_categorical(yCV, 10)
    # Build the model
    model = Sequential()
    model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(1,28,28)))
    model.add(Activation(activation))
    model.add(Convolution2D(32,3,3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16,3,3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
	# fit module
    print "fit module"
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit(Xtrain,ytrain,batch_size=100,nb_epoch=20,verbose=1,validation_data=(XCV,yCV))
    score = model.evaluate(Xtest,ytest, verbose=0)
    print score[0]
    print score[1]
if __name__ == "__main__":
	CNN_3_layer("relu")
