import os
import numpy

def run(dimension=50,shouldprint=True):
	if shouldprint:
		print 'Creatind GloVe dictionary'	
	dic = {}
	glovepath = "~/experiments/glove.6B.%sd.txt" % str(dimension)
	
	if shouldprint:
		print 'Loading GloVe vectors'
	for line in open(glovepath,'r').readlines():
		l = line.split()
		word = l[0]
		vet = [float(v) for v in l[1:]]		
		dic[word]=vet
	
	if shouldprint:
		print 'Done generating GloVe matrix'
	return dic

if __name__=='__main__':
	run(dimension=3)
