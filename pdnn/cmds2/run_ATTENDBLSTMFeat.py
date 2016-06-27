# Copyright 2015    Suyoun Kim    Carnegie Mellon University

import cPickle
import gzip
import os
import sys
import time

import numpy
import json

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_func.model_io import _file2nnet, log
from io_func.kaldi_feat import KaldiReadIn, KaldiWriteOut

from models.attendblstm import ATTEND_LSTM
from utils.utils import parse_arguments, string_2_bool

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['in_scp_file', 'out_ark_file', 'lstm_param_file', 'lstm_cfg_file']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    in_scp_file = arguments['in_scp_file']
    out_ark_file = arguments['out_ark_file']
    lstm_param_file = arguments['lstm_param_file']
    lstm_cfg_file = arguments['lstm_cfg_file']
    layer_index = int(arguments['layer_index'])

    # network structure
    cfg = cPickle.load(open(lstm_cfg_file,'r'))
    cfg.init_activation() 

    kaldiread = KaldiReadIn(in_scp_file)
    kaldiwrite = KaldiWriteOut(out_ark_file)

    log('> ... setting up the ATTEND LSTM layers')
    rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    lstm = ATTEND_LSTM(numpy_rng=rng, theano_rng=theano_rng, cfg = cfg)
    _file2nnet(layers = lstm.layers, set_layer_num = len(lstm.layers), filename=lstm_param_file)
    out_function = lstm.build_extract_feat_function()

    while True:
        uttid, in_matrix = kaldiread.read_next_utt()
        if uttid == '':
            break
        print 'in_matrix:'+str(in_matrix.shape)
        final_matrix = numpy.zeros((in_matrix.shape[0],cfg.n_outs), dtype=theano.config.floatX)
        remainder = in_matrix.shape[0]%cfg.batch_size
        for index in xrange(in_matrix.shape[0]/cfg.batch_size):
            final_matrix[index*cfg.batch_size:(index+1)*cfg.batch_size] = out_function(in_matrix[index*cfg.batch_size:(index+1)*cfg.batch_size])
        if remainder > 0:
            print '\tremainder:'+str(remainder)
            final_matrix[-remainder:] = out_function(in_matrix[-remainder:])
        print 'final_matrix:'+str(final_matrix.shape)
        kaldiwrite.write_kaldi_mat(uttid, final_matrix)

    kaldiwrite.close()

    log('> ... the saved features are %s' % (out_ark_file))
