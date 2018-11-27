#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle
import sys

reload(sys)  
sys.setdefaultencoding('utf-8')

def puct(sent):
	punctation = ['.',',','\'','\"','<','>',';',':','?','/','!']
	for p in punctation:
		sent = sent.replace(p,' %s '%p)
	return sent

filename = sys.argv[1]
stop = open('stopwords.txt','r').readlines()

qopair = cPickle.load(open(filename, "rb" ))
vocab = dict()


for p in qopair:
	words = puct(p[0]).lower().split()
	for w in words:
		if w not in stop:
			vocab[w] = vocab.get(w,0)+1
		

import operator
sorted_x = sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)

print sorted_x[0:200]