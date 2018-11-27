from keras.layers import Input, Dense, merge, Embedding, Lambda, Dropout
from keras.layers import Convolution1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import numpy as np
from abc import abstractmethod
import os
import random
import cPickle
import sys
import loademb
import copy
import time
import importlib

reload(sys)  
sys.setdefaultencoding('utf-8')


#### Defines ####

writefolder = '~/experiments/oncoqa_cnnlstm/'

#### Base class ####
class LanguageModel:
	def __init__(self, config, emb_weights=None):
		self.question = Input(shape=(config['question_len'],), dtype='int32', name='question_base')
		self.answer_good = Input(shape=(config['answer_len'],), dtype='int32', name='answer_good_base')
		self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32', name='answer_bad_base')
		self.config = config
		self.params = config.get('similarity', dict())

		# initialize a bunch of variables that will be set later
		self._models = None
		self._answer = None
		self._qa_model = None
		self._emb_weights=emb_weights

		self.training_model = None
		self.prediction_model = None

	def get_answer(self):
		if self._answer is None:
			self._answer = Input(shape=(self.config['answer_len'],), dtype='int32', name='answer')
		return self._answer

	@abstractmethod
	def build(self):
		return

	def get_similarity(self):
		dot = lambda a, b: K.batch_dot(a, b, axes=1)
		return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

	def get_qa_model(self):
		if self._models is None:
			self._models = self.build()

		if self._qa_model is None:
			question_output, answer_output = self._models
			dropout = Dropout(self.params.get('dropout', 0.2))
			similarity = self.get_similarity()
			qa_model = merge([dropout(question_output), dropout(answer_output)],
							 mode=similarity, output_shape=lambda _: (None, 1))
			self._qa_model = Model(input=[self.question, self.get_answer()], output=qa_model, name='qa_model')

		return self._qa_model
		
	def compile(self, optimizer, **kwargs):
		qa_model = self.get_qa_model()

		good_similarity = qa_model([self.question, self.answer_good])
		bad_similarity = qa_model([self.question, self.answer_bad])

		loss = merge([good_similarity, bad_similarity],
					 mode=lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
					 output_shape=lambda x: x[0])

		self.prediction_model = Model(input=[self.question, self.answer_good], output=good_similarity, name='prediction_model')
		self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

		self.training_model = Model(input=[self.question, self.answer_good, self.answer_bad], output=loss, name='training_model')
		self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

	def fit(self, x, **kwargs):
		assert self.training_model is not None, 'Must compile the model before fitting data'
		y = np.zeros(shape=(x[0].shape[0],)) # doesn't get used
		return self.training_model.fit(x, y, **kwargs)

	def predict(self, x):
		#assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
		return self.prediction_model.predict_on_batch(x)

	def save_weights(self, file_name, **kwargs):
		assert self.prediction_model is not None, 'Must compile the model before saving weights'
		self.prediction_model.save_weights(file_name, **kwargs)

	def load_weights(self, file_name, **kwargs):
		assert self.prediction_model is not None, 'Must compile the model loading weights'
		self.prediction_model.load_weights(file_name, **kwargs)


class QAModel(LanguageModel):
	def setEmbLayer(self):
		# add embedding layers
		if not len(self._emb_weights)>0:
			self._emb_weights = np.zeros((self.config['nb_words'] + 1, self.config['emb_dimension']))
	
		embedding = Embedding(input_dim=len(self._emb_weights),
						  output_dim=self._emb_weights.shape[1],
						  weights=[self._emb_weights],
						  trainable=1,
						  name='EmbeddingLayer')
		self.question_model = embedding(self.bquestion)
		self.answer_model = embedding(self.banswer)
	
	def setConvLayer(self,layersize=1000,layertrainable=1):
		cnns = [Convolution1D(filter_length=filter_length,
			         	nb_filter=layersize/4,
					activation='tanh',
					border_mode='same',
				  	trainable=layertrainable,
				  	name='CNN%d-kernel%d_Layer%d'%(layersize,filter_length,self.layerindex)) for filter_length in [2, 3, 5, 7]]
		question_cnn = merge([cnn(self.question_model) for cnn in cnns], mode='concat')
		answer_cnn = merge([cnn(self.answer_model) for cnn in cnns], mode='concat')
		
		self.question_model = self.dropout(question_cnn)#.reshape()
		self.answer_model = self.dropout(answer_cnn)#.reshape()

		self.layerindex+=1
	
	def setLSTMLayer(self,layersize=100,layertrainable=1):
		bilstm = Bidirectional(LSTM(layersize, input_length=self.config['question_len'],trainable=layertrainable,
					return_sequences=True,name='biLSTM%d_Layer%d'%(layersize,self.layerindex)), merge_mode="sum")
		
		self.question_model = self.dropout(bilstm(self.question_model))
		self.answer_model = self.dropout(bilstm(self.answer_model))
		self.layerindex+=1
	
	def setOutLayer(self):
		# maxpooling
		maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
		maxpool.supports_masking = True
		enc = Dense(100, activation='tanh')
		self.question_pool = self.dropout(enc(maxpool(self.question_model)))
		self.answer_pool = self.dropout(enc(maxpool(self.answer_model)))
	
	def build(self):
		self.bquestion = self.question
		self.banswer = self.get_answer()
		self.dropout = Dropout(self.params.get('dropout', 0.05))
		
		self.question_pool = None
		self.question_model = None
		self.answer_pool = None
		self.answer_model = None
		
		self.setEmbLayer()
		self.layerindex=1

		for layerparams in self.config['architecture']:
			if layerparams[0]=='CNN':
				self.setConvLayer(layerparams[1],layerparams[2])
			elif layerparams[0]=='LSTM':
				self.setLSTMLayer(layerparams[1],layerparams[2])

		self.setOutLayer()

		return self.question_pool, self.answer_pool

def pad(data, leng=None):
	from keras.preprocessing.sequence import pad_sequences
	return pad_sequences(data, maxlen=leng, padding='post', truncating='post', value=0)
	
def sortVets (qvet, avet):
    print len(avet), len(qvet)

    qvet_aux = copy.copy(qvet)
    avet_aux = copy.copy(avet)
    index_vet = [i for i in range(len(qvet))]
    random.shuffle(index_vet)

    for i in range(len(index_vet)):
        qvet_aux[i]=qvet[index_vet[i]]
        avet_aux[i]=avet[index_vet[i]]
    
    return qvet_aux, avet_aux

##### Loading / saving #####

def save_epoch(model, epoch):
    if not os.path.exists(writefolder+'models/'):
        os.makedirs(writefolder+'models/')
    model.save_weights(writefolder+'models/weights_epoch_%d.h5' % epoch, overwrite=True)

def load_epoch(model,epoch,custompath = None):
    if not custompath:
    	path = writefolder+'models/weights_epoch_%d.h5' % epoch
    else:
		path = custompath+'models/weights_epoch_%d.h5' % epoch
    print path
    assert os.path.exists(path), 'Weights at epoch %d not found' % epoch
    model.load_weights(path)
    return model

def delete_epoch(epoch):
    print 'Deleting '+writefolder+'models/weights_epoch_%d.h5' % epoch
    if os.path.isfile(writefolder+'models/weights_epoch_%d.h5' % epoch):
		os.remove(writefolder+'models/weights_epoch_%d.h5' % epoch)

def getEpoch(path):
	epoch = None
	for i in range(1,999): #Search for a file with weights among epochs (only one exists)
		if os.path.isfile('%smodels/weights_epoch_%d.h5'%(path,i)):
			epoch=i
	return epoch
	
def loadPickle(inp):
	with open(inp, 'rb') as fid:
    		clf= cPickle.load(fid)
	return clf
##### Process #####

def processData(QAS, EMBEDDING_DIM=50,embfile="90data.txt",returnmat=True):
	#Create label to id dictionary
	embDic = loademb.run(EMBEDDING_DIM,not returnmat, embfile)
	
	texts = [k.encode('utf8') for k in embDic.keys()]
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	if returnmat:
		print('Found %s unique tokens.' % len(word_index))

	questions = []
	answers = []
	text = ''
	
	#Get textual entries
	for qa in  QAS:
		qvec = text_to_word_sequence(''.join(c.encode('utf-8') for c in qa[0].lower()))
		for i in range(len(qvec)):
			qvec[i] = word_index.get(qvec[i],0)
		questions.append(qvec)
	
		avec = text_to_word_sequence(''.join(c.encode('utf-8') for c in qa[1].lower()))
		for i in range(len(avec)):
			avec[i] = word_index.get(avec[i],0)
		answers.append(avec)
	
	if not returnmat:
		if len(questions)==0 or len(answers)==0:
			return []
		return questions,answers

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embDic.get(word,None)
		if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
	embedding_matrix[0] = [1.e-17]*EMBEDDING_DIM
	
	return questions, answers, embedding_matrix

	
def train(model,params,answers,quests,modelname,deleteold=False,valdata=[]):
	batch_size = params['batch_size']
	nb_epoch = params['nb_epoch']
	validation_split = params['validation_split']

	print 'Training %s'%writefolder

	questions = pad(quests,params.get('question_len', None))
	answers_pos = pad(answers,params.get('answer_len', None))

	if len(valdata)>0:
		qval = pad(valdata[0],params.get('question_len', None))
		aval = pad(valdata[1],params.get('answer_len', None))

	#Sort data for better validation
	questions, answers_pos = sortVets(questions,answers_pos)

	val_loss = {'loss': 1., 'epoch': 0}	
	log = ""
	#Train one epoch at a time
	for i in range(1, nb_epoch+1):
		print('Fitting epoch %d' % i)
		#Train against X negative answers samples at each epoch. Select the largest loss for update
		neg_loss = {'loss': -1, 'hist':  None}
		for j in range(0,params['neg_examples']):
			print '\tNegative samples %d of %d'%(j+1,params['neg_examples'])
			model.save_weights(writefolder+'models/temp.h5', overwrite=True)
			answers_neg = pad(random.sample(answers_pos, len(answers_pos)),params.get('answer_len', None))
			
			if len(valdata)==0 or valdata==[]:
				print '\t...Fitting model. Validation with %.1f%% data'%(validation_split*100)
				hist = model.fit([questions, answers_pos, answers_neg], nb_epoch=1, batch_size=batch_size,
							 validation_split=validation_split, verbose=1)
			else:
				nval = pad(random.sample(aval, len(aval)),params.get('answer_len', None))
				print '\t...Fitting model. Validation with validation dataset'
				hist = model.fit([questions, answers_pos, answers_neg], nb_epoch=1, batch_size=batch_size,
							 validation_data=([qval,aval,nval],np.array([1]*len(aval))), verbose=1)
							 
			if hist.history['val_loss']	> neg_loss['loss']:
				model.save_weights(writefolder+'models/loss.h5', overwrite=True)
				neg_loss = {'loss':hist.history['val_loss'], 'hist':copy.copy(hist)}
							 
			model.load_weights(writefolder+'models/temp.h5')
		model.load_weights(writefolder+'models/loss.h5')  #Select the model with the largest loss for weights update			 
		
		if neg_loss['hist'].history['val_loss'][0] < val_loss['loss']: #Delete old best epoch
			old_best = val_loss['epoch']
			val_loss = {'loss': neg_loss['hist'].history['val_loss'][0], 'epoch': i}
			save_epoch(model,i)
			delete_epoch(old_best)

		log+=('-- Epoch %d\t' % ( i) +
			'Loss = %.4f, Validation Loss = %.4f ' % (hist.history['loss'][0], hist.history['val_loss'][0]) +
			'(Best: Loss = %.4f, Epoch = %d)\n' % (val_loss['loss'], val_loss['epoch']))
	outlog = open(writefolder+modelname+'_trainlog.txt','w')
	outlog.write(log)
	outlog.close()
	os.remove(writefolder+'models/temp.h5')
	os.remove(writefolder+'models/loss.h5')
	return model
	
	
def evaluate(model,params,answers,questions,datatype=None):
	i=0
	total = 0.0
	correct = 0.0
	top5correct = 0.0
	predlog = ""
	lasttime = 0
	lastpercent = 0.0
	predstart = 0
	predend = 0

	answers_padded = []
	questions_padded = []

	for a in range(len(answers)):
		answers_padded.append(pad([answers[a]],params.get('answer_len',None)))
		questions_padded.append(pad([questions[a]],params.get('question_len',None)))

        starttime = time.time()

	print 'Evaluating pre-trained model %s. %d Answers'%(writefolder.split('/')[-1],len(answers))
	for i in range(len(answers)):
		now = time.time()
		if int(now)>lasttime and total>0:
			percent = float(int(((i+1)*1000000)/(float(len(answers)))))/10000.0
			now = time.time()
			totaltime = now - starttime
			elapsed = totaltime
			predtime = 100.0*totaltime/percent
			remtime = predtime - elapsed
			remhour = int(remtime/3600)
			remmins = int(remtime/60) - remhour*60
			remsecs = remtime - remmins*60 - remhour*3600


			sys.stdout.write('\rProgress %.3f%% %d/%d (acc:%.3f)(%dh:%02dm:%02ds) [Last: %.2fs]' %(percent,i+1,len(answers),correct/total,remhour,remmins,remsecs,predend-predstart))
			sys.stdout.flush()			
			lastpercent = percent                	
			lasttime = int(now)
		answer_good = answers_padded[i]
		qst_good = questions_padded[i]
		answers_bad = list()

		for k in range(1,11):
			answers_bad.append((answers_padded[i-k],i-k))                       

		for k in range(1,11):
			pos = i+k - (int((i+k)/len(answers))*len(answers))
			answers_bad.append((answers_padded[pos],pos))
  
		predstart = time.time()

		answer_good = model.predict([qst_good,answer_good])
		
		for k in range(len(answers_bad)):
                        answers_bad[k] = (model.predict([qst_good,answers_bad[k][0]]),answers_bad[k][1])

		predend = time.time()
		
		total+=1
		incorrect=False
		thistop5=0
		predline='%f\t'%(answer_good)
		for bad in answers_bad:
			if questions[bad[1]]!=questions[i]:
				predline+='%f\t'%(bad[0])
				if bad[0]>answer_good:
					thistop5+=1
		top5correct+=int(thistop5<5) #This means that there are less than 5 answers which are better then the correct one    
		correct+=int(not thistop5) #This means that no answer is better than the correct one
			
		predlog+='%d'%(int(not thistop5))+predline+'\n'

	print '\n\nOverall\tACC: %f' %(correct/total)
	print 'Top5\tACC: %f' %(top5correct/total)
	if datatype==None:
		reslog = open((writefolder+'result_epoch_%d.txt' % epoch),'w')
		reslog.write('Overall\tACC: %f\t(%d/%d)\nTop5\tACC: %f\t(%d/%d)' %(correct/total,correct,total,top5correct/total,top5correct,total))
		reslog.close()
		predfile = open((writefolder+'predictions_epoch_%d.txt' % epoch),'w')
		predfile.write(predlog)
		predfile.close()
	else:
		reslog = open((writefolder+'result_epoch_%d_%s.txt' % (epoch,datatype)),'w')
		reslog.write('Overall\tACC: %f\t(%d/%d)\nTop5\tACC: %f\t(%d/%d)' %(correct/total,correct,total,top5correct/total,top5correct,total))
		reslog.close()
		predfile = open((writefolder+'predictions_epoch_%d_%s.txt' % (epoch,datatype)),'w')
		predfile.write(predlog)
		predfile.close()
		

def fullevaluate(model,params,answers,questions,datatype=None):
        i=0
        total = 0.0
        correct = 0.0
        top5correct = 0.0
        predlog = ""
        lasttime = 0
        lastpercent = 0.0
        predstart = 0
        predend = 0

        answers_padded = []
        questions_padded = []

        for a in range(len(answers)):
                answers_padded.append(pad([answers[a]],params.get('answer_len',None)))
                questions_padded.append(pad([questions[a]],params.get('question_len',None)))

        starttime = time.time()
	lasttime = time.time()
        print 'Evaluating pre-trained model %s. %d Answers'%(writefolder.split('/')[-1],len(answers))
        for i in range(len(answers)):
                now = time.time()
                if int(now)>lasttime and total>0:
                        percent = float(int(((i+1)*1000000)/(float(len(answers)))))/10000.0
                        totaltime = now - starttime
                        elapsed = totaltime
                        #predtime = 100.0*totaltime/percent
                        #remtime = predtime - elapsed
                        remtime = (now-lasttime)*(len(answers)-i)
			remhour = int(remtime/3600)
                        remmins = int(remtime/60) - remhour*60
                        remsecs = remtime - remmins*60 - remhour*3600


                        sys.stdout.write('\rProgress %.3f%% %d/%d (acc:%.3f)(%dh:%02dm:%02ds) [Last: %.2fs]' %(percent,i+1,len(answers),correct/total,remhour,remmins,remsecs,predend-predstart))
                        sys.stdout.flush()
                        lastpercent = percent
                        lasttime = int(now)
                answer_good = answers_padded[i]
		qst_good = questions_padded[i]
                answers_bad = list()

                predstart = time.time()

                answer_good = model.predict([qst_good,answer_good])
                for k in range(len(answers_padded)):
			if k!=i:
                        	answers_bad.append((model.predict([qst_good,answers_padded[k]]),k))

                predend = time.time()

                total+=1
                incorrect=False
                thistop5=0
                predline='%f\t'%(answer_good)
                for bad in answers_bad:
                        if questions[bad[1]]!=questions[i]:
                                predline+='%f\t'%(bad[0])
                                if bad[0]>answer_good:
                                        thistop5+=1
                top5correct+=int(thistop5<5) #This means that there are less than 5 answers which are better then the correct one
                correct+=int(not thistop5) #This means that no answer is better than the correct one

                predlog+='%d'%(int(not thistop5))+predline+'\n'

        print '\n\nOverall\tACC: %f' %(correct/total)
        print 'Top5\tACC: %f' %(top5correct/total)
        if datatype==None:
                reslog = open((writefolder+'FULLeval-result_epoch_%d.txt' % epoch),'w')
                reslog.write('Overall\tACC: %f\t(%d/%d)\nTop5\tACC: %f\t(%d/%d)' %(correct/total,correct,total,top5correct/total,top5correct,total))
                reslog.close()
                predfile = open((writefolder+'FULLeval-predictions_epoch_%d.txt' % epoch),'w')
                predfile.write(predlog)
                predfile.close()
        else:
                reslog = open((writefolder+'FULLeval-result_epoch_%d_%s.txt' % (epoch,datatype)),'w')
                reslog.write('Overall\tACC: %f\t(%d/%d)\nTop5\tACC: %f\t(%d/%d)' %(correct/total,correct,total,top5correct/total,top5correct,total))
                reslog.close()
                predfile = open((writefolder+'FULLeval-predictions_epoch_%d_%s.txt' % (epoch,datatype)),'w')
                predfile.write(predlog)
                predfile.close()



def processFreezing(model,conf):
	print 'Freezing layers'
	i = 0
	for l in model.training_model.layers[3].layers:
		if conf['freeze'][i]==0:
			print '\tFroze %s'%l.name
			l.trainable=0
			i+=1
		else:
			print '\tTrainable %s'%l.name

	model.compile(conf['optimizer'])
	return model

	
if __name__=='__main__':

    '''conf = {
        'n_words': 20000,
        'question_len': 30,
        'answer_len': 30,
        'margin': 0.2,
        'optimizer':'adam',
        
        #'building': {
	    'emb_dimension' : 100,
	    'modelname': 'qa_ebm100-cnn250-cnn500-lstm100-maxp',
		'architecture': [['EMB',100,1],['CNN',250,1],['CNN',500,1],['LSTM',100,1],['OUT',300,1]],
		'freeze': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],       
		'dropout': 0.2,

        #},
        
        #'training': {
            'neg_examples': 5,
			'batch_size': 100,
            'nb_epoch': 100,
            'validation_split': 0.1,
        #}
    }
	'''
    mode = sys.argv[1]
    datapath = sys.argv[2]
    
    confname = sys.argv[3].split('/')[-1]
    confpath = sys.argv[3][0:len(sys.argv[3])-len(confname)]
    confname = confname.split('.')[0]

    sys.path.append(confpath)
    mod = importlib.import_module(confname)
    conf = mod.conf
    
    embfile = sys.argv[4]

    try:
        modelname=sys.argv[5]
    except:
        modelname = conf['modelname']

    global writefolder
    writefolder+=modelname+'/'

    wpath = writefolder.split('/')
    for i in range(0,len(wpath)):
        path = ''.join(k+'/' for k in wpath[:i])
        if not os.path.exists(path[:-1]) and len(path[:-1])>1:
            os.makedirs(path[:-1])
    if not os.path.exists(writefolder+'models'):
	os.makedirs(writefolder+'models')

    questions, answers, embedding_matrix = processData(loadPickle(datapath),conf['emb_dimension'],embfile)
    print 'INICIANDO TREINAMENTO'
    if mode=='-train':
			model = QAModel(conf, embedding_matrix)
			model.compile(conf['optimizer'])
			model = train(model,conf,answers,questions,modelname,deleteold=True)
    elif mode=='-eval':
	domain = 'ALL'
	epoch = getEpoch(writefolder)
	print 'Evaluating model found at epoch %d\n'%epoch
	model = QAModel(conf, embedding_matrix)
        model.compile(conf['optimizer'])
        
        model = processFreezing(model,conf)
	model = load_epoch(model,epoch)

	evaluate(model,conf,answers,questions,domain)
    elif mode=='-feval':
        domain = 'ALL'
        epoch = getEpoch(writefolder)
        print 'Evaluating model found at epoch %d\n with all answers as candidates'%epoch
        model = QAModel(conf, embedding_matrix)
        model.compile(conf['optimizer'])

        model = processFreezing(model,conf)
        model = load_epoch(model,epoch)

        fullevaluate(model,conf,answers,questions,domain)


