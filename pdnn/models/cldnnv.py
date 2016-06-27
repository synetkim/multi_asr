# Copyright 2015    Suyoun Kim    Carnegie Mellon University

import cPickle
import gzip
import os
import sys
import time
import collections

import numpy
import json

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, OutputLayer
from layers.lstm import LSTMLayer, LSTMLayerForward
from layers.conv import ConvLayer, ConvLayerForward
from dnnv import DNNV

class CLDNNV(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, testing = False, input = None):

        self.cfg = cfg
        self.params = []
        self.delta_params   = []
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.l1_reg = cfg.l1_reg
        self.l2_reg = cfg.l2_reg
        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size
        self.max_col_norm = cfg.max_col_norm

        self.layers = []
        self.conv_layers = []
        self.lstm_layers = []
        self.fc_layers = []

        # 1. conv 
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.conv_layers_number = len(self.conv_layer_configs)
        self.use_fast = cfg.use_fast
        # 2. lstm
        self.lstm_layers_sizes = cfg.lstm_layers_sizes
        self.lstm_layers_number = len(self.lstm_layers_sizes)
        # 3. dnn
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        self.activation = cfg.activation

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if input == None:
            self.x = T.matrix('x')
        else:
            self.x = input
        self.y = T.matrix('y') 
       
        #######################
        # build conv layers   #
        #######################
        print '1. start to build conv layer: '+ str(self.conv_layers_number)
        for i in xrange(self.conv_layers_number):
            if i == 0:
                input = self.x
            else:
                input = self.conv_layers[-1].output
            config = self.conv_layer_configs[i]
            conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,
								input_shape = config['input_shape'], 
								filter_shape = config['filter_shape'], 
								poolsize = config['poolsize'],
								activation = self.conv_activation, 
								flatten = config['flatten'], 
								use_fast = self.use_fast, testing = testing)
            print '\tbuild conv layer: ' +str(config['input_shape'])
            self.layers.append(conv_layer)
            self.conv_layers.append(conv_layer)
            self.params.extend(conv_layer.params)
            self.delta_params.extend(conv_layer.delta_params)

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        print '\t cnn out: '+ str(self.conv_output_dim)
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        print '1. finish conv layer: '+ str(self.layers[-1].n_out)

        #######################
        # build lstm layers   #
        #######################
        print '2. start to build lstm layer: '+ str(self.lstm_layers_number)
        for i in xrange(self.lstm_layers_number):
            if i == 0:
                input_size = self.conv_output_dim 
                input = self.layers[-1].output
            else:
                input_size = self.lstm_layers_sizes[i - 1]
                input = self.layers[-1].output
            print 'build lstm layer: ' + str(input_size)
            lstm_layer = LSTMLayer(rng=numpy_rng, input=input, n_in=input_size, n_out=self.lstm_layers_sizes[i])
            print '\tbuild lstm layer: ' + str(input_size) +' x '+ str(lstm_layer.n_out)
            self.layers.append(lstm_layer)
            self.lstm_layers.append(lstm_layer)
            self.params.extend(lstm_layer.params)
            self.delta_params.extend(lstm_layer.delta_params)
        print '2. finish lstm layer: '+ str(self.layers[-1].n_out)

        #######################
        # build dnnv layers   #
        #######################
        print '3. start to build dnnv layer: '+ str(self.hidden_layers_number)
        for i in xrange(self.hidden_layers_number):
            if i == 0:
                input_size = self.layers[-1].n_out
            else:
                input_size = self.hidden_layers_sizes[i - 1]
            input = self.layers[-1].output
            fc_layer = HiddenLayer(rng=numpy_rng, input=input, n_in=input_size, n_out=self.hidden_layers_sizes[i])
            print '\tbuild dnnv layer: ' + str(input_size) +' x '+ str(fc_layer.n_out)
            self.layers.append(fc_layer)
            self.fc_layers.append(fc_layer)
            self.params.extend(fc_layer.params)
            self.delta_params.extend(fc_layer.delta_params)
        print '3. finish dnnv layer: '+ str(self.layers[-1].n_out)

        #######################
        # build log layers   #
        #######################
        print '4. start to build log layer: 1'
        input_size = self.layers[-1].n_out
        input = self.layers[-1].output
        logLayer = OutputLayer(input=input, n_in=input_size, n_out=self.n_outs)
        print '\tbuild final layer: ' + str(input_size) +' x '+ str(fc_layer.n_out)
        self.layers.append(logLayer)
        self.params.extend(logLayer.params)
        self.delta_params.extend(logLayer.delta_params)
        print '4. finish log layer: '+ str(self.layers[-1].n_out)
        print 'Total layers: '+ str(len(self.layers))

        sys.stdout.flush()

        self.finetune_cost = self.layers[-1].l2(self.y)
        self.errors = self.layers[-1].errors(self.y)

        if self.l2_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
		self.y: train_set_y[index * batch_size:
			            (index + 1) * batch_size]})

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
		self.y: valid_set_y[index * batch_size:
			            (index + 1) * batch_size]})

        return train_fn, valid_fn

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')
        out_da = theano.function([feat], self.layers[output_layer].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da


