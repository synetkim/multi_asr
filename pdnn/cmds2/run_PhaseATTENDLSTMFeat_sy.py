# Copyright 2014    Yajie Miao    Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

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

from models.cnnv import CNNV, CNN_Forward
from models.dnnv import DNNV
from models.phaseattendlstm import PhaseATTEND_LSTM, PhaseATTENDLSTM_Forward
from utils.utils import parse_arguments, string_2_bool

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['extra_in_scp_file', 'in_scp_file', 'out_ark_file', 'lstm_param_file', 'lstm_cfg_file']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    in_scp_file = arguments['in_scp_file']
    out_ark_file = arguments['out_ark_file']
    extra_in_scp_file = arguments['extra_in_scp_file']
    lstm_param_file = arguments['lstm_param_file']
    lstm_cfg_file = arguments['lstm_cfg_file']
    layer_index = int(arguments['layer_index'])

    # network structure
    cfg = cPickle.load(open(lstm_cfg_file,'r'))

    kaldiread = KaldiReadIn(in_scp_file)
    extra_kaldiread = KaldiReadIn(extra_in_scp_file)
    kaldiwrite = KaldiWriteOut(out_ark_file)

    log('> ... setting up the ATTEND LSTM layers')
    rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    cfg.init_activation() 
    lstm = PhaseATTENDLSTM_Forward(numpy_rng=rng, lstm_layer_configs = cfg, n_ins = cfg.n_ins)
    _file2nnet(layers = lstm.lstm_layers, set_layer_num = lstm.lstm_layer_num, filename=lstm_param_file)
    out_function = lstm.build_out_function()

    log('> ... setting up the DNN layers')
    dnn = DNNV(numpy_rng = rng, theano_rng = theano_rng, cfg = cfg, input=lstm.lstm_layers[-1].output)
    _file2nnet(layers = dnn.layers, set_layer_num = len(dnn.layers)+lstm.lstm_layer_num, filename = lstm_param_file, start_layer = lstm.lstm_layer_num)
    out_function2 = dnn.build_extract_feat_function(layer_index)

    log('> ... processing the data')

    while True:
        uttid, in_matrix = kaldiread.read_next_utt()
        extra_uttid, extra_in_matrix = extra_kaldiread.read_next_utt()
        if uttid == '':
            break

        print 'batch:' + str(cfg.batch_size)
        final_matrix = numpy.zeros((in_matrix.shape[0],cfg.n_outs), dtype=theano.config.floatX)
        for index in xrange(in_matrix.shape[0]/cfg.batch_size):
            mid_matrix = out_function(in_matrix[index*cfg.batch_size:(index+1)*cfg.batch_size])
            final_matrix[index*cfg.batch_size:(index+1)*cfg.batch_size] = out_function2(mid_matrix)
        print final_matrix.shape
        kaldiwrite.write_kaldi_mat(uttid, final_matrix)

    kaldiwrite.close()

    log('> ... the saved features are %s' % (out_ark_file))
