import cPickle
import sys
import random

file1 = cPickle.load(open(sys.argv[1], "rb" ))
file2 = cPickle.load(open(sys.argv[2],'rb'))

name = sys.argv[3]
data = file1

repeated = 0

print 'Merging files with %d and %d instances (expected dataset size: %d)'%(len(file1),len(file2),len(file1)+len(file2))

for f in file2:
	if f not in file1:
		data.append(f)
	else:
		repeated+=1

print 'Found %d instances of replicated data. Dataset size: %d'%(repeated,len(data))

#ASKME
if sys.argv[4] == '-r':
	print '\tIntitializing seed %d'%int(sys.argv[3][-1])
	random.seed(int(sys.argv[3][-1]))

	print '\tRandomizing dataset'
	random.shuffle(data)




cPickle.dump( data, open("datasets/all-%s.pkl"%name, "wb" ) )

print 'Creating train dataset. Size: %d'%len(data[:int(0.9*len(data))])	
cPickle.dump(data[:int(0.9*len(data))], open( "datasets/train-%s.pkl"%name, "wb" ) )

print 'Creating dev dataset. Size: %d'%len(data[int(0.9*len(data)):])
cPickle.dump( data[int(0.9*len(data)):], open( "datasets/dev-%s.pkl"%name, "wb" ) )

#print data[int(0.9*len(data)):]

