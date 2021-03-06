# Copyright 2015    Suyoun Kim    Carnegie Mellon University

import cPickle
import gzip
import os
import sys
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.attendlstm import ATTEND_LSTM
from models.phaseattendlstm import PhaseATTEND_LSTM

from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate

from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'lstm_nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    nnet_spec = arguments['nnet_spec']
    lstm_nnet_spec = arguments['lstm_nnet_spec']
    wdir = arguments['wdir']

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig();cfg.model_type = 'ATTEND_LSTM'
    cfg.parse_config_attend(arguments, nnet_spec, lstm_nnet_spec)
    cfg.init_data_reading(train_data_spec, valid_data_spec)
    print 'Extra dim: '+str(cfg.extra_dim)

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # setup model
    dnn = PhaseATTEND_LSTM(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y), 
                (cfg.extra_train_x), (cfg.extra_valid_x),
                batch_size=cfg.batch_size)

    log('> ... finetuning the model')
    min_verr = 100
    while (cfg.lrate.get_rate() != 0):
        train_error = train_sgd(train_fn, cfg)
        log('> epoch %d, training error %0.4f ' % (cfg.lrate.epoch, 100*numpy.mean(train_error)) )
        valid_error = validate_by_minibatch(valid_fn, cfg)
        log('> epoch %d, lrate %f, validation error %0.4f ' % (cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(valid_error)))
        cfg.lrate.get_next_rate(current_error = 100*numpy.mean(valid_error))
        if min_verr > 100*numpy.mean(valid_error):
            min_verr = 100*numpy.mean(valid_error)
            _nnet2file(dnn.layers, filename=cfg.param_output_file )
            _lrate2file(cfg.lrate, wdir + '/training_state.tmp') 

    if cfg.cfg_output_file != '':
        _cfg2file(dnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

