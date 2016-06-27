# Copyright 2015    Suyoun Kim    Carnegie Mellon University

import cPickle
import gzip
import os
import sys
import time
import collections

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer, OutputLayer
from layers.lstm import LSTMLayer
from layers.attendlstm import AttendLSTMLayer, SUMLayer

from io_func.model_io import _nnet2file, _file2nnet

class ATTEND_LSTM(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None, draw=None):

        self.cfg = cfg
        self.params = []
        self.delta_params   = []
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.l1_reg = cfg.l1_reg
        self.l2_reg = cfg.l2_reg
        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size
        self.max_col_norm = 1
        print self.max_col_norm

        self.layers = []
        self.lstm_layers = []
        self.fc_layers = []
        self.bilayers = []

        # 1. lstm
        self.lstm_layers_sizes = cfg.lstm_layers_sizes
        self.lstm_layers_number = len(self.lstm_layers_sizes)
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
        self.y = T.ivector('y')

        #######################
        # build lstm layers   #
        #######################
        print '1. start to build AttendLSTMLayer : '+ str(self.lstm_layers_number) + ', n_attendout: '+ str(cfg.batch_size)
        for i in xrange(1):
            if i == 0:
                input_size = self.n_ins
                input = self.x
            else:
                input_size = self.lstm_layers_sizes[i - 1]
                input = self.bilayers[-1].output

            # Forward
            f_lstm_layer = AttendLSTMLayer(rng=numpy_rng, input=input, n_in=input_size, n_out=self.lstm_layers_sizes[i],
                                                steps=cfg.batch_size, draw=draw)
            print '\tbuild f_lstm layer: ' + str(input_size) +' x '+ str(f_lstm_layer.n_out)
            self.layers.append(f_lstm_layer)
            self.lstm_layers.append(f_lstm_layer)
            self.params.extend(f_lstm_layer.params)
            self.delta_params.extend(f_lstm_layer.delta_params)

            # Backward
            b_lstm_layer = AttendLSTMLayer(rng=numpy_rng, input=input, n_in=input_size, n_out=self.lstm_layers_sizes[i], backwards=True,
                                                steps=cfg.batch_size, draw=draw)
            print '\tbuild b_lstm layer: ' + str(input_size) +' x '+ str(b_lstm_layer.n_out)
            self.layers.append(b_lstm_layer)
            self.lstm_layers.append(b_lstm_layer)
            self.params.extend(b_lstm_layer.params)
            self.delta_params.extend(b_lstm_layer.delta_params)

            # Sum forward + backward
            bi_layer = SUMLayer(finput=f_lstm_layer.output,binput=b_lstm_layer.output[::-1], n_out=self.lstm_layers_sizes[i - 1])
            self.bilayers.append(bi_layer)
            print '\tbuild sum layer: ' + str(input_size) +' x '+ str(bi_layer.n_out)

        print '1. finish AttendLSTMLayer: '+ str(self.bilayers[-1].n_out)

        #######################
        # build log layers   #
        #######################
        print '3. start to build log layer: 1'
        input_size = self.bilayers[-1].n_out
        input = self.bilayers[-1].output
        logLayer = LogisticRegression(input=input, n_in=input_size, n_out=self.n_outs)
        print '\tbuild final layer: ' + str(input_size) +' x '+ str(self.n_outs)
        self.layers.append(logLayer)
        self.params.extend(logLayer.params)
        self.delta_params.extend(logLayer.delta_params)
        print '3. finish log layer: '+ str(self.bilayers[-1].n_out)
        print 'Total layers: '+ str(len(self.layers))

        sys.stdout.flush()

        self.finetune_cost = logLayer.negative_log_likelihood(self.y)
        self.errors = logLayer.errors(self.y)

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

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.fc_layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
            for i in xrange(self.lstm_layers_number):
                todo_W_list = [ self.lstm_layers[i].W_xi,
                                self.lstm_layers[i].W_hi,
                                self.lstm_layers[i].W_xf,
                                self.lstm_layers[i].W_hf,
                                self.lstm_layers[i].W_xc,
                                self.lstm_layers[i].W_hc,
                                self.lstm_layers[i].W_xo,
                                self.lstm_layers[i].W_ho,
                                self.lstm_layers[i].W_hy,
                              ]
                for todo_W in todo_W_list:
                    W = todo_W
                    if W in updates:
                        print 'Gradient Clipping!' + str(self.max_col_norm)
                        updated_W = updates[W]
                        col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                        desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                        updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

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


    def build_extract_feat_function(self):
        feat = T.matrix('feat')
        out_da = theano.function([feat], self.layers[-1].output, 
                                  updates = None, 
                                  givens={self.x:feat}, 
                                  on_unused_input='warn')
        return out_da

    def write_model_to_raw(self, file_path):
        # output the model to tmp_path; this format is readable by PDNN
        _nnet2file(self.layers, filename=file_path)


