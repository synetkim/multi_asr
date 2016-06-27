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

import theano
from theano import tensor as T, function, printing
from theano.tensor.shared_randomstreams import RandomStreams

class AttendRnnLayer(object):

    def __init__(self, rng, input, n_in, n_out, n_attendout, initial_hidden=None, W_rec = None, activation=T.tanh):
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.n_attendout = n_attendout
        self.n_attendin = 100

        self.type = 'attendrnn'
        
        if initial_hidden is None:
            initial_hidden_values_s = numpy.zeros((n_out,), dtype=theano.config.floatX)
            initial_hidden_s = theano.shared(value=initial_hidden_values_s, name='s0', borrow=True)
        self.s0 = initial_hidden_s

        if W_rec is None:
            W_type1 = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W_type2 = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_out)), dtype=theano.config.floatX)
            W_type3 = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(self.n_attendin, self.n_attendout)), dtype=theano.config.floatX)
            W_type4 = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(self.n_in, self.n_attendin)), dtype=theano.config.floatX)
            W_type5 = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(self.n_out, self.n_attendin)), dtype=theano.config.floatX)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

            W_ic = theano.shared(value=W_type1, name='W_ic', borrow=True)
            W_rec = theano.shared(value=W_type2, name='W_rec', borrow=True)
            W_outattend = theano.shared(value=W_type3, name='W_outattend', borrow=True)
            W_inattend_feat = theano.shared(value=W_type4, name='W_inattend_feat', borrow=True)
            W_inattend_prevstate = theano.shared(value=W_type5, name='W_inattend_prevstate', borrow=True)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W_ic = W_ic
        self.W_rec = W_rec
        self.W_outattend = W_outattend
        self.W_inattend_feat = W_inattend_feat
        self.W_inattend_prevstate = W_inattend_prevstate
        self.b = b

        self.delta_W_ic = theano.shared(value = numpy.zeros((n_in,n_out),
                                     dtype=theano.config.floatX), name='delta_W_ic')
        self.delta_W_rec = theano.shared(value = numpy.zeros((n_out,n_out),
                                     dtype=theano.config.floatX), name='delta_W_rec')
        self.delta_W_outattend = theano.shared(value = numpy.zeros((self.n_attendin,self.n_attendout),
                                     dtype=theano.config.floatX), name='delta_W_outattend')
        self.delta_W_inattend_feat = theano.shared(value = numpy.zeros((n_in,self.n_attendin),
                                     dtype=theano.config.floatX), name='delta_W_inattend_feat')
        self.delta_W_inattend_prevstate = theano.shared(value = numpy.zeros((n_out,self.n_attendin),
                                     dtype=theano.config.floatX), name='delta_W_inattend_prevstate')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
                                     dtype=theano.config.floatX), name='delta_b')

        self.test8 = numpy.zeros((8,), dtype=theano.config.floatX)

        # sequences: h_l
        # prior results: s_tm1 
        # non sequences: W_outattend, W_inattend_prevstate, W_ic, W_rec, b, W_inattend_feat
        def one_step(h_l, s_tm1, W_outattend, W_inattend_prevstate, W_ic, W_rec, b, W_inattend_feat):
            e_tl = T.dot(T.tanh(T.dot(s_tm1,W_inattend_prevstate) + T.dot(h_l,W_inattend_feat)), W_outattend) 
            a_tl = T.exp(e_tl)/(T.exp(e_tl)).sum(0,keepdims=True)
            c_t = T.dot(a_tl,self.input)
            s_t = T.tanh(T.dot(c_t, W_ic) + T.dot(s_tm1, W_rec) + b) 
            return s_t

        self.y_vals, _ = theano.scan(fn=one_step, 
											sequences=self.input, 
											outputs_info=self.s0,
											non_sequences = [self.W_outattend, self.W_inattend_prevstate, self.W_ic, self.W_rec, self.b, self.W_inattend_feat])

        # parameters of the model
        self.params = [self.W_outattend, self.W_inattend_prevstate, self.W_ic, self.W_rec, self.b, self.W_inattend_feat]
        self.delta_params = [self.delta_W_outattend, self.delta_W_inattend_prevstate, self.delta_W_ic, self.delta_W_rec, self.delta_b, self.delta_W_inattend_feat]

        sigma = lambda x: 1 / (1 + T.exp(-x)) 
        self.output = sigma(self.y_vals)

    def xent(self, target):
        self.cost = -T.mean(target * T.log(self.y_vals)+ (1.- target) * T.log(1. - self.y_vals))
        return self.cost

    def l2(self, target):
        error = T.mean((self.y_vals - target)**2)
        return error



