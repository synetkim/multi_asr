
import sys
import os
import cPickle, gzip
import numpy as np
from sklearn import preprocessing

data = sys.stdin
fout1 = sys.argv[1]
n_feat = int(sys.argv[2])
n_label = int(sys.argv[3])
print "outfile test: " + fout1

rawdata = np.loadtxt(data, delimiter=" ")

test_data = rawdata
#print "zero mean unit variance"
#test_data = preprocessing.scale(rawdata)

nrow = len(test_data[:,1])
print "test #instances: "+str(nrow)
f_indices=np.arange(2,2+n_feat)
l_indices=np.arange(2+n_feat,2+n_feat+n_label)
cPickle.dump((test_data[:,f_indices].astype(np.float32), test_data[:,l_indices].astype(np.float32)), gzip.open(fout1,'wb'), cPickle.HIGHEST_PROTOCOL)
