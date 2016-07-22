# -*- coding:utf-8 -*-
#########################################################################
# File Name: classify.py
# Author: libanghuai
# mail: libanghuai@pku.edu.cn
# Created Time: 2016/06/12 21:38:18 CST
#########################################################################

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input
import jieba
import itertools
import numpy as np
import re
import copy
from copy import deepcopy
import unicodedata
from gensim.models import Word2Vec
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

if __name__ == '__main__':

	#Data Processing
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
			'Data/train',
			target_size=(150, 150),
			batch_size=32,
			class_mode='categorical')

	test_datagen = ImageDataGenerator(rescale=1./255)

	validation_generator = test_datagen.flow_from_directory(
			'Data/validation',
			target_size=(150, 150),
			batch_size=32,
			class_mode="categorical")

	model = Sequential()
    # this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 150, 150)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(13))
	model.add(Activation('softmax'))
	
	model.compile(loss='categorical_crossentropy',
			optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
			metrics=['accuracy'])
        
	model.fit_generator(
		train_generator,
		samples_per_epoch=977,
		nb_epoch=20,
		validation_data=validation_generator,
		nb_val_samples=250)

