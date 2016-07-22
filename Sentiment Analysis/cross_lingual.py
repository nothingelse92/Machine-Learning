#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# File Name: cross_lingual.py
# Author: libanghuai
# mail: libanghuai@pku.edu.cn
# Created Time: 2016/04/21 20:29:10 CST
#########################################################################

import re
import urllib
import urllib2
import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import SVC

def translate(text,ls,ts):
	#Translate the text from language ls to language ts
	values={'hl':'zh-CN','ie':'UTF-8','text':text,'langpair':"%s|%s" % (ls,ts)}
	url='http://translate.google.cn/'
	data = urllib.urlencode(values)  
	req = urllib2.Request(url,data)
	browser='Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)'
	req.add_header('User-Agent',browser)
	response = urllib2.urlopen(req)
	html=response.read()
	p=re.compile(r"(?<=TRANSLATED_TEXT=).*?;")
	m=p.search(html)
	ans=m.group(0).strip(';')
	return ans 

def parser(file_path):
	# Get the summary and text of the review
	open_file = open(file_path)
	file_lines = open_file.read()
	while True:
		summary = ''
		text = ''
		pattern = re.compile(r'<summary>')
		match_objs = pattern.search(file_lines)
		if match_objs:
			s1 = match_objs.start()
			pattern = re.compile(r'</summary>')
			temp_objs = pattern.search(file_lines)
			if temp_objs:
				t1 = temp_objs.start()
				summary = file_lines[(s1+9):t1]
				file_lines = file_lines[(t1+10):]  
			else:
				break
		else:
			break
		pattern = re.compile(r'<text>')
		match_objs = pattern.search(file_lines)
		if match_objs:
			s1 = match_objs.start()
			pattern = re.compile(r'</text>')
			temp_objs = pattern.search(file_lines)
			if temp_objs:
				t1 = temp_objs.start()
				text = file_lines[(s1+6):t1]
				file_lines = file_lines[(t1+7):]  
			else:
				break
		else:
			break
		yield summary+' '+text

def get_labels(file_path):
	# Get the corresponding labels of these reviews
	open_file = open(file_path)
	file_lines = open_file.read()
	while True:
		pattern = re.compile(r'<polarity>')
		match_objs = pattern.search(file_lines)
		if match_objs:
			s1 = match_objs.start()
			pattern = re.compile(r'</polarity>')
			temp_objs = pattern.search(file_lines)
			if temp_objs:
				t1 = temp_objs.start()
				label = file_lines[(s1+10):t1]
				file_lines = file_lines[(t1+11):]  
				num = 1
				if label == 'N' or label == 'n':
					num = -1
				yield num
			else:
				break
		else:
			break

#分词操作
def word_segment(docs):
	token_doc = []
	for doc in docs:
		token_doc.append(list(jieba.cut(doc)))
	return token_doc

#TF_IDF model
def tfidf_array(cor):
	corpus=[]
	for cc in cor:
		ss = ""
		for ccc in cc:
			ss += ccc
		corpus.append(ss)
	vectorizer=CountVectorizer()
	transformer=TfidfTransformer()
	tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
	word=vectorizer.get_feature_names()
	X=tfidf.toarray()
	print X[0]
	return X

#SVM Model
def my_svm(X,Y):
	clf = svm.SVC()
	clf.fit(X,Y)
	return clf

#Predict some docments' labels
def predict_docs(clf,docs):
	file_ans = open('ans','wb')
	ans = []
	for doc in docs:
		docc = []
		docc.append(doc)
		temp_ans = clf.predict(docc)
		ans.append(temp_ans)
		file_ans.write(str(temp_ans) + '\n')
	return ans

if __name__=="__main__":
	file_path_train = 'train.data'
	file_path_test = 'test.data'
	docs = list(parser(file_path_train))
	len_en = len(docs)
	labels = list(get_labels(file_path_train))
	docs_cn = list(parser(file_path_test))
	for doc in docs_cn:
		docs.append(translate(doc,'zh-CN','en'))
	len_all = len(docs)
	token_docs = word_segment(docs)
	doc_array = tfidf_array(token_docs)
	model = my_svm(doc_array[0:len_en],labels[0:len_en])
	ans = predict_docs(model,doc_array[len_en:len_all])
