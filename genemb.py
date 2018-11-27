# !/usr/bin/python
# -*- coding: utf-8  -*-

from gensim.models import Word2Vec
import sys
import cPickle

reload(sys)  
sys.setdefaultencoding('utf-8')

stopwords = open('stopwords.txt','r').read().split()

def puct(sent):
	punctation = ['.',',','\'','\"','<','>',';',':','?','/','!','(',')','\\']
	for p in punctation:
		sent = sent.replace(p,' %s '%p)
	return sent

print 'Loading data'

file1 = cPickle.load(open(sys.argv[1], "rb" ))
data = list()

for i in range(0,int(0.9*len(file1))):
	data.append(file1[i][0])
	data.append(file1[i][1])

try:
	file2 = cPickle.load(open(sys.argv[2],'rb'))
	for i in range(0,int(0.9*len(file2))):
		data.append(file2[i][0])
		data.append(file2[i][1])
except:
	pass

print 'Data loaded. %d instances found'%len(data)
	
print 'Removing stopwords'	
removed = 0
for i in range(0,len(data)):
	sys.stdout.write('\r\t Processed %d instances.'%(i))
	splitsentence = puct(data[i]).lower().split()
	
	#for w in range(0,len(splitsentence)):
		#if splitsentence[w] in stopwords:
		#	splitsentence[w]='---XXX---'
	#data[i]=''
	#for w in splitsentence:
	#	if w!='---XXX---':
	#		data[i]+=w+' '
	#data[i] = data[i][:-1]
	data[i] = splitsentence
		

			
print '\nTraining model'
		
model = Word2Vec(data, size=200, window=5, min_count=2, workers=4,iter=100)

print 'Saving model'

model.save('datasets/90emb-%s.emb'%sys.argv[1].split('/')[-1])


print '---Done---'

print 'Saving model in %s'%('datasets/90emb-%s.txt'%sys.argv[1].split('/')[-1])

txtout = open('datasets/90emb-%s.txt'%(sys.argv[1].split('/')[-1]),'w')

for key in model.wv.vocab.keys():
	txtout.write(key)
	#print len(model.wv[key])
	#raw_input()
	for val in model.wv[key]:
		txtout.write('\t%f'%val)
	txtout.write('\n')

txtout.close()

