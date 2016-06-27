import sys
import cPickle
import gzip
import numpy as np

f=gzip.open(sys.argv[1], 'rb')

feat = cPickle.load(f)
print "feat shape: " +str(feat.shape)
