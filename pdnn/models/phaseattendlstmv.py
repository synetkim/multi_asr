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
from layers.phaseattendlstm import PhaseAttendLSTMLayer

from io_func.model_io import _nnet2file, _file2nnet

class PhaseATTEND_LSTM(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None, extra_input = None):

        self.cfg = cfg
        self.params = []
        self.delta_params   = []
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.l1_reg = cfg.l1_reg
        self.l2_reg = cfg.l2_reg
        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size
        self.max_col_norm = cfg.max_col_norm
        print self.max_col_norm

        self.layers = []
        self.extra_layers = []
        self.lstm_layers = []
        self.fc_layers = []

        # 1. lstm
        self.lstm_layers_sizes = cfg.lstm_layers_sizes
        self.lstm_layers_number = len(self.lstm_layers_sizes)
       
        # 1.5 attention
        self.extra_dim = cfg.extra_dim
        print 'Extra dim: '+str(cfg.extra_dim)
        self.extra_layers_sizes = cfg.extra_layers_sizes

        # 2. dnn
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        self.activation = cfg.activation


        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if input == None:
            self.x = T.matrix('x')
            self.extra_x = T.matrix('extra_x')
        else:
            self.x = input 
            self.extra_x = extra_input 
        self.y = T.matrix('y')

        #######################################
        # build phase-based attention layer   #
        #######################################
        # 0. phase-based attention
        #self.extra_layers_sizes.extend([self.conv_output_dim])
        #print '0. start to build attend layer: '+ str(self.extra_layers_sizes)
        #for i in xrange(len(self.extra_layers_sizes)):
        #    if i == 0:
        #        input_size = 6400*5
        #        input_size = cfg.extra_dim
        #        layer_input = self.extra_x
        #    else:
        #        input_size = self.extra_layers_sizes[i - 1]
        #        layer_input = self.extra_layers[-1].output
#
#            W = None; b = None
#            attend_layer = HiddenLayer(rng=numpy_rng,
#                                        input=layer_input,
#                                        n_in=input_size,
#                                        n_out=self.extra_layers_sizes[i],
#                                        W = W, b = b,
#                                        activation=self.activation)
#            print '\tbuild attend layer: ' + str(input_size) +' x '+ str(attend_layer.n_out)
#            self.extra_layers.append(attend_layer)
#            self.params.extend(attend_layer.params)
#            self.delta_params.extend(attend_layer.delta_params)
#        self.extra_layers[-1].att_e_tl = self.extra_layers[-1].output
#        self.extra_layers[-1].att_a_tl = T.nnet.softmax(self.extra_layers[-1].att_e_tl)
#        #self.extra_layers[-1].att_a_tl = T.exp(self.extra_layers[-1].att_e_tl)/(T.exp(self.extra_layers[-1].att_e_tl)).sum(0,keepdims=True)
#        print '0. finish attend layer: '+ str(self.extra_layers[-1].n_out)

        #######################
        # build lstm layers   #
        #######################
        #print '1. start to build PhaseAttendLSTMLayer : '+ str(self.lstm_layers_number) + ', n_attendout: '+ str(cfg.batch_size)
        print '1. start to build PhaseAttendLSTMLayer : '+ str(self.lstm_layers_number) + ', n_attendout: '+ str(self.n_ins)
        for i in xrange(self.lstm_layers_number):
            if i == 0:
                input_size = self.n_ins
                input = self.x
            else:
                input_size = self.lstm_layers_sizes[i - 1]
                input = self.layers[-1].output
            lstm_layer = PhaseAttendLSTMLayer(rng=numpy_rng, input=input, n_in=input_size, 
											extra_input = extra_input, 
											n_out=self.lstm_layers_sizes[i])
            print '\tbuild PhaseAttendLSTMLayer: ' + str(input_size) +' x '+ str(lstm_layer.n_out)
            self.layers.append(lstm_layer)
            self.lstm_layers.append(lstm_layer)
            self.params.extend(lstm_layer.params)
            self.delta_params.extend(lstm_layer.delta_params)
        print '1. finish PhaseAttendLSTMLayer: '+ str(self.layers[-1].n_out)

        #######################
        # build dnnv layers   #
        #######################
        print '2. start to build dnnv layer: '+ str(self.hidden_layers_number)
        for i in xrange(self.hidden_layers_number):
            if i == 0:
                input_size = self.layers[-1].n_out
            else:
                input_size = self.hidden_layers_sizes[i - 1]
            input = self.layers[-1].output
            fc_layer = HiddenLayer(rng=numpy_rng, input=input, n_in=input_size, n_out=self.hidden_layers_sizes[i], activation=self.activation)
            print '\tbuild dnnv layer: ' + str(input_size) +' x '+ str(fc_layer.n_out)
            self.layers.append(fc_layer)
            self.fc_layers.append(fc_layer)
            self.params.extend(fc_layer.params)
            self.delta_params.extend(fc_layer.delta_params)
        print '2. finish dnnv layer: '+ str(self.layers[-1].n_out)

        #######################
        # build log layers   #
        #######################
        print '3. start to build log layer: 1'
        input_size = self.layers[-1].n_out
        input = self.layers[-1].output
        logLayer = OutputLayer(input=input, n_in=input_size, n_out=self.n_outs)
        print '\tbuild final layer: ' + str(input_size) +' x '+ str(fc_layer.n_out)
        self.layers.append(logLayer)
        self.params.extend(logLayer.params)
        self.delta_params.extend(logLayer.delta_params)
        print '3. finish log layer: '+ str(self.layers[-1].n_out)
        print 'Total layers: '+ str(len(self.layers))

        sys.stdout.flush()

        self.finetune_cost = self.layers[-1].l2(self.y)
        self.errors = self.layers[-1].errors(self.y)

        if self.l2_reg is not None:
            #for i in xrange(self.lstm_layers_number):
            #    W = self.lstm_layers[i].W_xi
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_hi
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_xf
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_hf
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_xc
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_hc
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_xo
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            #    W = self.lstm_layers[i].W_ho
            #    self.finetune_cost += self.l2_reg * T.sqr(W).sum()
            for i in xrange(self.hidden_layers_number):
                W = self.fc_layers[i].W
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()


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
								self.lstm_layers[i].W_ho]
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


    #def build_extract_feat_function(self, output_layer):
#
#        feat = T.matrix('feat')
#        out_da = theano.function([feat], self.layers[output_layer].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
#        return out_da

    def write_model_to_raw(self, file_path):
        # output the model to tmp_path; this format is readable by PDNN
        _nnet2file(self.layers, filename=file_path)


class PhaseATTENDLSTM_Forward(object):
    def __init__(self, numpy_rng, cfg, n_ins):
        self.lstm_layers = []
        self.extra_layers = []
        self.extra_layers_sizes = cfg.extra_layers_sizes
        self.lstm_layer_num = len(lstm_layer_configs.lstm_layers_sizes)
        self.extra_layers_sizes = cfg.extra_layers_sizes

        self.x = T.matrix('x')
        self.extra_x = T.matrix('extra_x')

        for i in xrange(self.lstm_layer_num):
            if i == 0:
                input = self.x
            else:
                input = self.lstm_layers[-1].output

            lstm_layer = PhaseAttendLSTMLayer(rng=numpy_rng, input=input,
		                                n_in=n_ins,
                                        n_out=lstm_layer_configs.lstm_layers_sizes[i],
       									extra_input = extra_input, 
										n_attendout=n_ins)
            self.lstm_layers.append(lstm_layer)

    def build_out_function(self):
        feat = T.matrix('feat')
        extra_feat = T.matrix('extra_feat')
        out_da = theano.function([feat, theano.Param(extra_feat)], 
								self.lstm_layers[-1].output, 
								updates = None, 
								givens={
									self.x: feat,
									self.extra_x: extra_feat
								}, 
								on_unused_input='warn')
        return out_da




