from gensim.models import Word2Vec
import sys

if (len(sys.argv) < 1):
  print "pass the word to check similarity"
  sys.exit(-1)
model = Word2Vec.load('90data.emb')
print model.wv.most_similar(sys.argv[1])