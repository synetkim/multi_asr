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
from models.dnnv import DNNV, DNN_Forward
from models.lstmv import LSTMV, LSTM_Forward
from models.cldnnv import CLDNNV
from utils.utils import parse_arguments, string_2_bool

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['in_scp_file', 'out_ark_file', 'cldnn_param_file', 'cldnn_cfg_file']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    in_scp_file = arguments['in_scp_file']
    out_ark_file = arguments['out_ark_file']
    cldnn_param_file = arguments['cldnn_param_file']
    cldnn_cfg_file = arguments['cldnn_cfg_file']
    layer_index = int(arguments['layer_index'])

    # network structure
    cfg = cPickle.load(open(cldnn_cfg_file,'r'))

    kaldiread = KaldiReadIn(in_scp_file)
    kaldiwrite = KaldiWriteOut(out_ark_file)

    rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    log('> ... setting up the CNN layers')
    conv_configs = cfg.conv_layer_configs 
    conv_layer_number = len(conv_configs) 
    use_fast = cfg.use_fast 
    input_shape_train = conv_configs[0]['input_shape']
    input_shape_1 = (input_shape_train[1], input_shape_train[2], input_shape_train[3])
    for i in xrange(conv_layer_number):
        conv_configs[i]['activation'] = cfg.conv_activation
    cfg.init_activation() 
    cnn = CNN_Forward(numpy_rng = rng, theano_rng=theano_rng, conv_layer_configs = conv_configs, use_fast = use_fast)
    _file2nnet(layers = cnn.conv_layers, set_layer_num = len(conv_configs), filename = cldnn_param_file, start_layer = 0)
    out_function1 = cnn.build_out_function()
    conv_output_dim = cfg.conv_layer_configs[-1]['output_shape'][1] * cfg.conv_layer_configs[-1]['output_shape'][2] * cfg.conv_layer_configs[-1]['output_shape'][3]
    print conv_output_dim 

    log('> ... setting up the LSTM layers')
    cfg.init_activation() 
    lstm = LSTM_Forward(lstm_layer_configs = cfg, n_ins = conv_output_dim)
    _file2nnet(layers = lstm.lstm_layers, set_layer_num = lstm.lstm_layer_num+len(conv_configs), filename = cldnn_param_file, start_layer = len(conv_configs))
    out_function2 = lstm.build_out_function()
    cfg.n_ins = cfg.lstm_layers_sizes[-1]

    log('> ... setting up the DNN layers')
    dnn = DNNV(numpy_rng = rng, theano_rng = theano_rng, cfg = cfg, input=lstm.lstm_layers[-1].output) 
    _file2nnet(layers = dnn.layers, set_layer_num = len(dnn.layers)+lstm.lstm_layer_num+len(conv_configs), filename = cldnn_param_file, start_layer = lstm.lstm_layer_num+len(conv_configs))
    out_function3 = dnn.build_extract_feat_function(layer_index)

    log('> ... processing the data')

    while True:
        uttid, in_matrix = kaldiread.read_next_utt()
        if uttid == '':
            break
        in_matrix = numpy.reshape(in_matrix, (in_matrix.shape[0],) + input_shape_1)
        tmp_matrix1 = out_function1(in_matrix)
        tmp_matrix2 = out_function2(tmp_matrix1)
        out_matrix = out_function3(tmp_matrix2)
        kaldiwrite.write_kaldi_mat(uttid, out_matrix)

    kaldiwrite.close()

    log('> ... the saved features are %s' % (out_ark_file))
