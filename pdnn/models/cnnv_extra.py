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
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, OutputLayer

from layers.conv import ConvLayer, ConvLayerForward
from dnnv import DNNV

class CNNV(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, testing = False, input = None):

        self.layers = []
        self.extra_layers = []
        self.params = []
        self.delta_params   = []
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.conv_layers = []

        self.cfg = cfg
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.use_fast = cfg.use_fast

        self.extra_x = T.matrix('extra_x')

        # 1.5 attention
        self.extra_dim = cfg.extra_dim
        print 'Extra input dimension: '+str(cfg.extra_dim)
        self.extra_layers_sizes = cfg.extra_layers_sizes

        # 2. dnn
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
        # build cnn layers   #
        ####################### 
        print '1. start to build cnn mag layer: '+ str(self.conv_layer_configs) 
        self.conv_layer_num = len(self.conv_layer_configs)
        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.layers[-1].output
            config = self.conv_layer_configs[i]
            conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,
			input_shape = config['input_shape'], filter_shape = config['filter_shape'], poolsize = config['poolsize'],
			activation = self.conv_activation, flatten = config['flatten'], use_fast = self.use_fast, testing = testing)
	    self.layers.append(conv_layer)
            self.conv_layers.append(conv_layer)
	    self.params.extend(conv_layer.params)
            self.delta_params.extend(conv_layer.delta_params)

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]

        #######################################
        # build phase-based attention layer   #
        #######################################
        # 0. phase-based attention
        print '2. start to build attend layer: '+ str(self.extra_layers_sizes)
        for i in xrange(len(self.extra_layers_sizes)):
            if i == 0:
                input_size = cfg.extra_dim
                layer_input = self.extra_x
            else:
                input_size = self.extra_layers_sizes[i - 1]
                layer_input = self.extra_layers[-1].output

            W = None; b = None
            attend_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.extra_layers_sizes[i],
                                        W = W, b = b,
                                        activation=self.activation)
            print '\tbuild attend layer: ' + str(input_size) +' x '+ str(attend_layer.n_out)
            self.extra_layers.append(attend_layer)
            self.params.extend(attend_layer.params)
            self.delta_params.extend(attend_layer.delta_params)
        self.extra_output = self.extra_layers[-1].output
        self.extra_output = T.nnet.softmax(self.extra_layers[-1].output)

        #self.extra_output_rand = numpy.asarray(numpy_rng.uniform(
        #            low=-0.1,
        #            high=1.0,
        #            size=(32,20)), dtype=theano.config.floatX)
        #self.extra_output = theano.shared(value=self.extra_output_rand, name='rand', borrow=True)
        print '2. finish attend layer softmax(0): '+ str(self.extra_layers[-1].n_out)
        #######################################
        # build dnnv                          #
        #######################################

        print '3. start to build dnnv layer: '+ str(self.hidden_layers_number)
        for i in xrange(self.hidden_layers_number):
            # construct the hidden layer
            if i == 0:
                # 1. Join two features (magnitude + phase)
                input_size = self.conv_output_dim + self.extra_layers_sizes[-1]
                layer_input = T.join(1, self.layers[-1].output, self.extra_output)
                # 2. Weighted Sum (magnitude * phase)
                #input_size = self.conv_output_dim 
                #layer_input = self.layers[-1].output * self.extra_output
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = self.layers[-1].output

            W = None; b = None
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        W = W, b = b,
                                        activation=self.activation)
            print '\tbuild dnnv layer: ' + str(input_size) +' x '+ str(hidden_layer.n_out)
            # add the layer to our list of layers
            self.layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            self.delta_params.extend(hidden_layer.delta_params)
        print '3. finish dnnv layer: '+ str(self.layers[-1].n_out)

        #######################################
        # build logistic regression layer     #
        #######################################
        print '4. start to build log layer: 1'
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = OutputLayer(
                         input=self.layers[-1].output,
                         n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs)
        print '\tbuild final layer: ' + str(self.layers[-1].n_out) +' x '+ str(self.n_outs)

        self.layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)
        self.delta_params.extend(self.logLayer.delta_params)
        print '4. finish log layer: '+ str(self.layers[-1].n_out)
        print 'Total layers: '+ str(len(self.layers))

        self.finetune_cost = self.logLayer.l2(self.y)
        self.errors = self.logLayer.errors(self.y)

        sys.stdout.flush()

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    # output conv config to files
    def write_conv_config(self, file_path_prefix):
        for i in xrange(len(self.conv_layer_configs)):
            self.conv_layer_configs[i]['activation'] = self.cfg.conv_activation_text
            with open(file_path_prefix + '.' + str(i), 'wb') as fp:
                json.dump(self.conv_layer_configs[i], fp, indent=2, sort_keys = True)
                fp.flush()

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, extra_train_shared_x, extra_valid_shared_x, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy
        (extra_train_set_x) = extra_train_shared_x
        (extra_valid_set_x) = extra_valid_shared_x

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
			            (index + 1) * batch_size],
                self.extra_x: extra_train_set_x[index * batch_size:
                                    (index + 1) * batch_size]
                }, on_unused_input='ignore')

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
		self.y: valid_set_y[index * batch_size:
			            (index + 1) * batch_size],
                self.extra_x: extra_valid_set_x[index * batch_size:
                                    (index + 1) * batch_size]
                }, on_unused_input='ignore')

        return train_fn, valid_fn

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')
        out_da = theano.function([feat], self.layers[output_layer].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

class CNN_Forward(object):

    def __init__(self, numpy_rng = None, theano_rng=None,
                    cfg = None,
                    non_maximum_erasing = False, use_fast = False):

        self.n_outs = cfg.n_outs
        self.layers = []
        self.extra_layers = []
        self.conv_layer_num = len(cfg.conv_layer_configs)
        self.dnn_layer_num = len(cfg.hidden_layers_sizes)
        self.extra_layers_sizes = cfg.extra_layers_sizes

        self.x = T.tensor4('x')
        self.extra_x = T.matrix('extra_x')

        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.layers[-1].output
            config = cfg.conv_layer_configs[i]
            conv_layer = ConvLayerForward(numpy_rng=numpy_rng, input = input,
                        filter_shape = config['filter_shape'], poolsize = config['poolsize'],
                        activation = config['activation'], flatten = config['flatten'], use_fast = use_fast)
            self.layers.append(conv_layer)

        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]

        for i in xrange(len(self.extra_layers_sizes)):
            if i == 0:
                input_size = 6400*5
                input_size = cfg.extra_dim
                layer_input = self.extra_x
            else:
                input_size = self.extra_layers_sizes[i - 1]
                layer_input = self.extra_layers[-1].output
            W = None; b = None
            attend_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.extra_layers_sizes[i],
                                        W = W, b = b
                                        )
            self.extra_layers.append(attend_layer)

        self.extra_layers[-1].att_e_tl = self.extra_layers[-1].output
        self.extra_layers[-1].att_a_tl = T.nnet.softmax(self.extra_layers[-1].att_e_tl)
        #self.extra_layers[-1].att_a_tl = T.exp(self.extra_layers[-1].att_e_tl)/(T.exp(self.extra_layers[-1].att_e_tl)).sum(0,keepdims=True)

        for i in xrange(self.dnn_layer_num):
            if i == 0:
                #input_size = self.conv_output_dim
                #layer_input = (self.extra_layers[-1].att_a_tl*self.layers[-1].output)
                input_size = self.conv_output_dim + self.extra_layers_sizes[-1]
                layer_input = T.join(1,self.extra_layers[-1].att_a_tl,self.layers[-1].output)
            else:
                input_size = cfg.hidden_layers_sizes[i - 1]
                layer_input = self.layers[-1].output
            W = None; b = None
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=cfg.hidden_layers_sizes[i],
                                        W = W, b = b
                                        )
            self.layers.append(hidden_layer)

        logLayer = OutputLayer(
                         input=self.layers[-1].output,
                         n_in=cfg.hidden_layers_sizes[-1], n_out=self.n_outs)
        self.layers.append(logLayer)

    def build_out_function(self):

        feat = T.tensor4('feat')
        extra_feat = T.matrix('extra_feat')
        out_da = theano.function(inputs=[feat,theano.Param(extra_feat)],
                                outputs=self.layers[-1].output,
                                updates=None,
                                givens={
                                    self.x: feat,
                                    self.extra_x: extra_feat
                                },
                                on_unused_input='warn')
        return out_da




class CNN_Forward(object):

    def __init__(self, numpy_rng = None, theano_rng=None, cfg = [], non_maximum_erasing = False, use_fast = False):

        self.conv_layers = []
        self.n_outs = cfg.n_outs
        self.layers = []
        self.extra_layers = []
        self.conv_layer_num = len(cfg.conv_layer_configs)
        self.dnn_layer_num = len(cfg.hidden_layers_sizes)
        self.extra_layers_sizes = cfg.extra_layers_sizes

        self.x = T.tensor4('x')
        self.extra_x = T.matrix('extra_x')

        for i in xrange(self.conv_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.conv_layers[-1].output
            config = cfg.conv_layer_configs[i]
            print config['filter_shape']
            conv_layer = ConvLayerForward(numpy_rng=numpy_rng, input = input,
                        filter_shape = config['filter_shape'], poolsize = config['poolsize'],
                        activation = config['activation'], flatten = config['flatten'], use_fast = use_fast)
            self.layers.append(conv_layer)
            self.conv_layers.append(conv_layer)
        self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
        cfg.n_ins = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
   
        print self.conv_output_dim
        print cfg.n_ins
        print 'Extra input dimension: '+str(cfg.extra_dim)
        for i in xrange(len(self.extra_layers_sizes)):
            if i == 0:
                input_size = cfg.extra_dim
                layer_input = self.extra_x
            else:
                input_size = self.extra_layers_sizes[i - 1]
                layer_input = self.extra_layers[-1].output
            W = None; b = None
            attend_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.extra_layers_sizes[i],
                                        W = W, b = b
                                        )
            self.extra_layers.append(attend_layer)
        self.extra_output = self.extra_layers[-1].output
        self.extra_output = T.nnet.softmax(self.extra_layers[-1].output)

        print 'layer num: ' +str(len(self.layers)-1)
        for i in xrange(self.dnn_layer_num):
            if i == 0:
                # 1. Join two features (magnitude + phase)
                input_size = self.conv_output_dim + self.extra_layers_sizes[-1]
                layer_input = T.join(1, self.layers[-1].output, self.extra_output)
                # 2. Weighted Sum (magnitude * phase)
                #input_size = self.conv_output_dim 
                #layer_input = self.layers[-1].output * self.extra_output
            else:
                input_size = cfg.hidden_layers_sizes[i - 1]
                layer_input = self.layers[-1].output
            W = None; b = None
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=cfg.hidden_layers_sizes[i],
                                        W = W, b = b
                                        )
            self.layers.append(hidden_layer)

        print 'layer num: ' +str(len(self.layers)-1)
        logLayer = OutputLayer(
                         input=self.layers[-1].output,
                         n_in=cfg.hidden_layers_sizes[-1], n_out=self.n_outs)
        self.layers.append(logLayer)
        print 'layer num: ' +str(len(self.layers)-1)



    def build_out_function(self):

        feat = T.tensor4('feat')
        extra_feat = T.matrix('extra_feat')
        out_da = theano.function([feat, theano.Param(extra_feat)], self.layers[-1].output, updates = None, 
                               givens={self.x:feat, self.extra_x: extra_feat}, on_unused_input='warn')
        return out_da
