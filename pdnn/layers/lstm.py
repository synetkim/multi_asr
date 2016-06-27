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

class SUMLayer(object):
    def __init__(self, finput, binput, n_out):
        self.output = finput+binput
        self.n_out = n_out

class LSTMLayer(object):

    def __init__(self, rng, input, n_in, n_out, initial_hidden=None, 
                 activation=T.tanh, backwards=False, peepholes=False):
        dtype=theano.config.floatX
        if backwards == True:
            print "backwards = True"
            self.input = input[::-1]
        else:
            self.input = input 
        self.n_in = n_in
        self.n_out = n_out
        self.n_steps = input.shape[0]
        self.type = 'lstmv'

        n_hidden = n_i = n_c = n_o = n_f = n_out
        n_y = n_out

        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=dtype)
            for dx in xrange(sizeX):
                nlow=-numpy.sqrt(6. / (sizeX + sizeY))
                nhigh=numpy.sqrt(6. / (sizeX + sizeY))
                vals = numpy.random.uniform(low=nlow, high=nhigh,  size=(sizeY,))
                values[dx,:] = vals*4
            return values

        self.W_xi = theano.shared(sample_weights(n_in, n_i), borrow=True)  
        self.W_hi = theano.shared(sample_weights(n_hidden, n_i), borrow=True)  
        self.W_ci = theano.shared(sample_weights(n_c, n_i), borrow=True)  
        self.W_xf = theano.shared(sample_weights(n_in, n_f), borrow=True) 
        self.W_hf = theano.shared(sample_weights(n_hidden, n_f), borrow=True)
        self.W_cf = theano.shared(sample_weights(n_c, n_f), borrow=True)
        self.W_xc = theano.shared(sample_weights(n_in, n_c), borrow=True)  
        self.W_hc = theano.shared(sample_weights(n_hidden, n_c), borrow=True)
        self.W_xo = theano.shared(sample_weights(n_in, n_o), borrow=True)
        self.W_ho = theano.shared(sample_weights(n_hidden, n_o), borrow=True)
        self.W_co = theano.shared(sample_weights(n_c, n_o), borrow=True)
        self.W_hy = theano.shared(sample_weights(n_hidden, n_y), borrow=True)
        self.delta_W_hy = theano.shared(sample_weights(n_hidden, n_y), borrow=True)  
        self.delta_W_xi = theano.shared(sample_weights(n_in, n_i), borrow=True)  
        self.delta_W_ci = theano.shared(sample_weights(n_c, n_i), borrow=True)  
        self.delta_W_hi = theano.shared(sample_weights(n_hidden, n_i), borrow=True)  
        self.delta_W_xf = theano.shared(sample_weights(n_in, n_f), borrow=True) 
        self.delta_W_hf = theano.shared(sample_weights(n_hidden, n_f), borrow=True)
        self.delta_W_cf = theano.shared(sample_weights(n_c, n_f), borrow=True)
        self.delta_W_xc = theano.shared(sample_weights(n_in, n_c), borrow=True)  
        self.delta_W_hc = theano.shared(sample_weights(n_hidden, n_c), borrow=True)
        self.delta_W_xo = theano.shared(sample_weights(n_in, n_o), borrow=True)
        self.delta_W_ho = theano.shared(sample_weights(n_hidden, n_o), borrow=True)
        self.delta_W_co = theano.shared(sample_weights(n_c, n_o), borrow=True)

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)
        self.b_hy = theano.shared(value=numpy.zeros(n_y,dtype=dtype), borrow=True)
        self.delta_b_hy = theano.shared(value=numpy.zeros(n_y,dtype=dtype), borrow=True)
        self.b_i = theano.shared(value=numpy.zeros(n_i,dtype=dtype), borrow=True)
        self.delta_b_i = theano.shared(value=numpy.zeros(n_i,dtype=dtype), borrow=True)
        self.b_f = theano.shared(value=numpy.zeros(n_f,dtype=dtype), borrow=True)
        self.delta_b_f = theano.shared(value=numpy.zeros(n_f,dtype=dtype), borrow=True)
        self.b_c = theano.shared(value=numpy.zeros(n_c,dtype=dtype), borrow=True)
        self.delta_b_c = theano.shared(value=numpy.zeros(n_c,dtype=dtype), borrow=True)
        self.b_o = theano.shared(value=numpy.zeros(n_o,dtype=dtype), borrow=True)
        self.delta_b_o = theano.shared(value=numpy.zeros(n_o,dtype=dtype), borrow=True)

        def one_lstm_step(x_t, h_tm1, c_tm1, W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho
                           ):
            i_t = T.nnet.sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) )
            f_t = T.nnet.sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) )
            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) ) 
            o_t = T.nnet.sigmoid(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) ) 
            h_t = o_t * T.tanh(c_t)
            return [h_t, c_t]

        def one_lstm_step_peepholes(x_t, h_tm1, c_tm1, W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho, 
                           W_ci, W_cf, W_co, b_i, b_f, b_c, b_o):
            i_t = T.nnet.sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
            f_t = T.nnet.sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c) 
            o_t = T.nnet.sigmoid(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o) 
            h_t = o_t * T.tanh(c_t)
            return [h_t, c_t]

        if peepholes == True:
            print "peepholes = True"
            [self.h_vals, _], _ = theano.scan(fn=one_lstm_step_peepholes, sequences=self.input, 
											outputs_info=[self.h0, self.c0],
											non_sequences=[self.W_xi, self.W_hi, 
                                                           self.W_xf, self.W_hf, 
                                                           self.W_xc, self.W_hc, 
                                                           self.W_xo, self.W_ho, self.W_ci, self.W_cf, self.W_co,
                                                           ],
											n_steps=self.n_steps, strict=True, allow_gc=False)
            self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc, 
                        self.W_hc, self.W_xo, self.W_ho, self.W_hy, self.b_hy, 
                        self.W_ci, self.W_cf, self.W_co, ]
            self.delta_params = [self.delta_W_xi, self.delta_W_hi, self.delta_W_xf, 
                        self.delta_W_hf, self.delta_W_xc, self.delta_W_hc, 
                        self.delta_W_xo, self.delta_W_ho, self.delta_W_hy, self.delta_b_hy,
                        self.delta_W_ci, self.delta_W_cf, self.delta_W_co, 
                        ]

        else:
            print "peepholes = False"
            [self.h_vals, _], _ = theano.scan(fn=one_lstm_step, sequences=self.input, 
											outputs_info=[self.h0, self.c0],
											non_sequences=[self.W_xi, self.W_hi, 
                                                           self.W_xf, self.W_hf, 
                                                           self.W_xc, self.W_hc, 
                                                           self.W_xo, self.W_ho, 
                                                           ],
											n_steps=self.n_steps, strict=True, allow_gc=False)
            self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc, 
                        self.W_hc, self.W_xo, self.W_ho, self.W_hy, self.b_hy, 
                        ]
            self.delta_params = [self.delta_W_xi, self.delta_W_hi, self.delta_W_xf, 
                        self.delta_W_hf, self.delta_W_xc, self.delta_W_hc, 
                        self.delta_W_xo, self.delta_W_ho, self.delta_W_hy, self.delta_b_hy,
                        ]

        self.output = T.nnet.sigmoid(theano.dot(self.h_vals,self.W_hy)+self.b_hy)
        self.y_vals = self.output

    def l2(self, target):
        error = T.mean((self.y_vals - target)**2)
        return error

