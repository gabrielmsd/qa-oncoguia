import sys
import subprocess
import os

if len(sys.argv)>=5:

	fulldataset = sys.argv[1].split('-')[1:]
	dataset=''
	for f in fulldataset:
		dataset='-'+f

	config = sys.argv[2]
	modelname = sys.argv[3] # ASKME
	reps = int(sys.argv[4])

else:
	print "\targv[1]: <dataset name>\n\targv[2]: <config file>\n\targv[3]: <modelname>\n\targv[4]: <num repetitions>"
	sys.exit(-1)

# word embeddings
for r in range(0,reps):
	if not os.path.isfile("datasets/90emb-all%s.txt"%dataset):
		 subprocess.call('python genemb.py datasets/all%s'%dataset,shell=True)

	subprocess.call('python cnnlstm_onco.py -train datasets/train%s %s datasets/90emb-all%s.txt %s-rep%d'%(dataset,config,dataset,modelname,r),shell=True)
	subprocess.call('python cnnlstm_onco.py -eval datasets/dev%s %s datasets/90emb-all%s.txt %s-rep%d'%(dataset,config,dataset,modelname,r),shell=True)	

