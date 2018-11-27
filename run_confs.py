import sys
import subprocess

fulldataset = sys.argv[1]
configs = ['50']
modelname = sys.argv[2]
reps = sys.argv[3]
secondpy = sys.argv[4] #<run_exp.py> or <run_feval.py>

for conf in configs:
	config = "configfile_%s.py"%conf
	dataname = fulldataset.split('/')[-1].split('.')[0].split('all-')[1]

	print "python %s %s %s %s %s"%(secondpy,fulldataset, config, '%s-%s_%s'%(modelname,conf,dataname),reps)
	subprocess.call("python %s %s %s %s %s"%(secondpy,fulldataset, config, '%s-%s_%s'%(modelname,conf,dataname),reps),shell=True)

