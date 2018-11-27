import os
import numpy

def run(dimension=50,shouldprint=True, embfile="90data.txt"):
	if shouldprint:
		print 'Creatind dictionary'	
	dic = {}
	glovepath = embfile
	
	if shouldprint:
		print 'Loading word vectors'
	for line in open(glovepath,'r').readlines():
		l = line.split()
		word = l[0]
		vet = [float(v) for v in l[1:]]	
		assert len(vet) == dimension, 'Word vectors have a different dimension (%d) from expected (%d)'%(len(vet),dimension)	
		dic[word]=vet
	
	if shouldprint:
		for key,val in dic.items():
			print "%s => %s"%(key,val)
		print 'Done generating matrix'
	return dic

if __name__=='__main__':
	run(dimension=200)
