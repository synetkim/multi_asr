
import sys
import os
import cPickle, gzip
import numpy as np
from sklearn import preprocessing

data = sys.stdin
fout1 = sys.argv[1]
fout2 = sys.argv[2]
n_feat = int(sys.argv[3])
n_label = int(sys.argv[4])
f_ratio = float(sys.argv[5])
print "outfile train: " + fout1
print "outfile valid: " + fout2
print "ratio: " + sys.argv[5]

rawdata = np.loadtxt(data, delimiter=" ")

train_valid = rawdata
#print "zero mean unit variance"
#train_valid = preprocessing.scale(rawdata)


nrow = len(train_valid[:,1])
indices = np.arange(nrow)
np.random.shuffle(indices)
print "total #instances: "+str(len(indices))
train_indices = indices[0:int(nrow*f_ratio)]
valid_indices = indices[int(nrow*f_ratio):len(indices)]
print "train #instances: "+str(len(train_indices))
print "valid #instances: "+str(len(valid_indices))
train = train_valid[ train_indices, : ]
valid = train_valid[ valid_indices, : ]
f_indices=np.arange(2,2+n_feat)
l_indices=np.arange(2+n_feat,2+n_feat+n_label)
cPickle.dump((train[:,f_indices].astype(np.float32), train[:,l_indices].astype(np.float32)), gzip.open(fout1,'wb'), cPickle.HIGHEST_PROTOCOL)
cPickle.dump((valid[:,f_indices].astype(np.float32), valid[:,l_indices].astype(np.float32)), gzip.open(fout2,'wb'), cPickle.HIGHEST_PROTOCOL)
