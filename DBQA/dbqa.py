# -*- coding:utf-8 -*-  
#########################################################################
# File Name: dbqa.py
# Author: libanghuai
# mail: libanghuai@pku.edu.cn
# Created Time: 2016/06/18 10:50:11 CST
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


from gensim import corpora, models, similarities

max_len = 10000
max_feature = 200000 # the word appear in the data is no more than 2000
word_embedding_dims = 200
train_range = 15000 
CV_range = 181882 

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

# cleaner (order matters)
def clean(text):
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def parser(filePath):
	data = open(filePath,'rb')
	for line in data:
		items = line.split('\t')
		yield items
        
def get_wordlist(word_list):
    for word in word_list:
        items = list(jieba.cut(word))
        yield items
        
def get_doc(word_list):
	word_lists = list(get_wordlist(word_list))
	stopwords = [' ','，','。','？','《','》','！','、','“','”',':','；']
	j = 0
	for words in word_lists:
		temp_word = []
		for word in words:
			if word not in stopwords:
				temp_word.append(word)
		yield  TaggedDocument(temp_word,['sen_%s' % j])
		j += 1
	print 'all_docs %d' % j

temp_docs = []
def my_doc2vec(word_list):
	all_docs = list(get_doc(word_list))
	print all_docs[0]
	global temp_docs
	temp_docs = copy.deepcopy(all_docs)
	basemodel = Doc2Vec(size=100,window=10,min_count=5,workers=11,alpha=0.025,min_alpha=0.025)
	basemodel.build_vocab(all_docs)
	for epoch in range(10):
		np.random.shuffle(all_docs)
		basemodel.train(all_docs)
		basemodel.alpha -= 0.002
		basemodel.min_alpha = basemodel.alpha
	maze = []
	for i in range(0,len(all_docs)):
		maze.append(basemodel.docvecs['sen_%s' % i])
#	print all_docs[0]
	return maze

def buildModel(filePath):
	QAData = list(parser(filePath))
	Xtrain = []# split the sentence(including question and answer) into words
	Ytrain = []# the lable of the answer 0 or 1
	XCV = []
	YCV = []
	Xtest = []
	Ytest = []
	docList = []
	i = 0
	for data in QAData:
		QA = data[0]+data[1]
		docList.append(QA)
		if i < train_range : 
			Ytrain.append(data[2][0])#截取的字段结尾有'\r''\n'等字符,所以只取第一个
		elif i < CV_range :
			YCV.append(data[2][0])
		i += 1

	vec = my_doc2vec(docList)

	for j in range(i):
		if j < train_range:
			Xtrain.append(vec[j])
		elif  j < CV_range:
			XCV.append(vec[j])
		else :
			Xtest.append(vec[j])

	'''
	words = set(itertools.chain(*Xtrain))
	word2idx = dict((v, i) for i, v in enumerate(words))
	idx2word = list(words)
	to_idx = lambda x: [word2idx[word] for word in x]
	sentences_idx = [to_idx(sentence) for sentence in Xtrain]
	sentences_array = np.asarray(sentences_idx, dtype='int32')
	'''
	Ytrain = np.array(Ytrain)
	Ytrain = Ytrain.reshape(train_range,1)
	Xtrain = np.array(Xtrain)
	Xtrain = Xtrain.reshape(train_range,100)
	Xtrain = np.reshape(Xtrain, Xtrain.shape + (1,))

	YCV = np.array(YCV)
	YCV = YCV.reshape(CV_range - train_range,1)
	XCV = np.array(XCV)
	XCV = XCV.reshape(CV_range - train_range,100)
	XCV = np.reshape(XCV, XCV.shape + (1,))

	Xtest = np.array(Xtest)
	Xtest = Xtest.reshape(i - CV_range,100)
	Xtest = np.reshape(Xtest, Xtest.shape + (1,))


	print "Build Model ..."
	QAModel = Sequential()
	QAModel.add(LSTM(128, activation='sigmoid', inner_activation='hard_sigmoid',return_sequences=False,input_shape=Xtrain.shape[1:]))
	QAModel.add(Dropout(0.5))
	QAModel.add(Dense(1))
	QAModel.add(Activation('sigmoid'))
	QAModel.compile(loss='binary_crossentropy', optimizer='rmsprop')
	QAModel.fit(Xtrain,Ytrain, batch_size=100, nb_epoch=20,validation_data=(XCV,YCV))

	file_ans = open('ans.txt','w')
	predict_ans = QAModel.predict(Xtest)
	for item in predict_ans:
		file_ans.write(str(item[0]))
		file_ans.write('\n')


if __name__ == '__main__':
	filePath = 'nlpcc-iccpol-2016.dbqa.training-data'
	buildModel(filePath)

