#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import cPickle
import pandas as pd

reload(sys)  
sys.setdefaultencoding('utf-8')

filename = sys.argv[1]

sheet = xls = pd.ExcelFile(filename)
sheetX = xls.parse(0)

qc = sheetX['questionamento']
oc = sheetX['orientacao_dada']

# Visualizing data
# print '%s rows'%max(len(qc),len(oc)) 
# data_direitos_sociais = sheetX[sheetX["Tema"] == "Direitos Sociais"]
# print '%s rows where Tema = Direitos Sociais'%len(data_direitos_sociais)
# print data_direitos_sociais[['questionamento', 'orientacao_dada']].head()

qopair = list()


for i in range (0, max(len(qc),len(oc))):
	try:
		quest = qc[i].replace('\"','')
		orien = oc[i].replace('\"','')
			
		if len(quest)>1 and len(orien)>1:
			qopair.append((quest,orien))
		
	except:
		continue
cPickle.dump( qopair[:], open( "all-%s.pkl"%filename.split('.')[0], "wb" ) )
cPickle.dump( qopair[:int(0.9*(len(qopair)))], open( "train-%s.pkl"%filename.split('.')[0], "wb" ) )
cPickle.dump( qopair[:int(0.1*(len(qopair)))], open( "dev-%s.pkl"%filename.split('.')[0], "wb" ) )

