import numpy as np
import os
import sys
import cPickle

from StringIO import StringIO
import json

#import theano
#import theano.tensor as T

from datetime import datetime
# convert an array to a string
def array_2_string(array):
    str_out = StringIO()
    np.savetxt(str_out, array)
    return str_out.getvalue()

# convert a string to an array
def string_2_array(string):
    str_in = StringIO(string)
    return np.loadtxt(str_in)


filename=sys.argv[1]
dict_a=sys.argv[2]

def main():
    nnet_dict = {}
    with open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
        print '===================================='
        for key in nnet_dict.keys():
            print key + str(np.asarray(string_2_array(nnet_dict[key])).shape)
        print '===================================='
        print 'dict_a:'+dict_a
        W = np.asarray(string_2_array(nnet_dict[dict_a]))
        print str(W.shape)
        print str(W)
        #print str(W)
        #print 'shape:'+str(W.shape)
        #print 'sum(1):'+str(W.sum(axis=1).shape)
        #print 'sum(0):'+str(W.sum(axis=0).shape)

        #e_x=np.exp(W)/np.exp(W).sum(1, keepdims=True)
        #print 'e/e.sum(1):'+str(e_x.shape)
        #print 'e.sum(1)keepdim: '+str(e_x.sum(1,keepdims=True).shape)+str(e_x.sum(1))

if __name__=="__main__":
    main()
