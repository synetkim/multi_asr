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

class AttendLSTMLayer(object):

    def __init__(self, rng, input, n_in, n_out, extra_input=None, n_extra_in=0,
                 activation=T.tanh, backwards=False, steps=1, draw=None):

        if backwards == True:
            print "backwards = True"
            self.input = input[::-1]
        else:
            self.input = input

        if n_extra_in >0:
            print 'n_extra_in: '+str(n_extra_in)
            # for attention weights
            if extra_input == None:
                print ''
                print '@@@@@@@@@@@@@@@@@ extra_input == None @@@@@@@@@@@@@@'
                print ''
                self.extra_x = T.matrix('extra_x')
            else:
                if backwards == True:
                    print "backwards = True"
                    self.extra_x = extra_input[::-1]
                else:
                    print "backwards = False"
                    self.extra_x = extra_input
        
        self.n_in = n_in
        self.n_extra_in = n_extra_in
        self.n_lstmin = 2200
        self.n_lstmin = 1400
        self.n_out = n_out
        self.n_steps = input.shape[0]
        self.n_attendout = n_in

        self.type = 'attendlstm'

        dtype=theano.config.floatX

        # sequences: x_t
        # prior results: h_tm1, c_tm1
        # non-sequences: W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, 
        W_xi = W_hi = None 
        W_xf = W_hf = None 
        W_xc = W_hc = None 
        W_xo = W_ho = None 

        # initialize weights
        # i_t and o_t should be "open" or "closed"
        # f_t should be "open" (don't forget at the beginning of training)
        # we try to archive this by appropriate initialization of the corresponding biases 
        def sample_weights(sizeX, sizeY):
            values = numpy.ndarray([sizeX, sizeY], dtype=dtype)
            for dx in xrange(sizeX):
                nlow=-numpy.sqrt(6. / (sizeX + sizeY))
                nhigh=numpy.sqrt(6. / (sizeX + sizeY))
                vals = numpy.random.uniform(low=nlow, high=nhigh,  size=(sizeY,))
                values[dx,:] = vals*4
            return values          

        n_hidden = n_i = n_c = n_o = n_f = n_out
        n_y = n_out

        if W_xi is None:
            W_xi = theano.shared(sample_weights(self.n_lstmin, n_i))  
            W_hi = theano.shared(sample_weights(n_hidden, n_i))  
            W_xf = theano.shared(sample_weights(self.n_lstmin, n_f)) 
            W_hf = theano.shared(sample_weights(n_hidden, n_f))
            W_xc = theano.shared(sample_weights(self.n_lstmin, n_c))  
            W_hc = theano.shared(sample_weights(n_hidden, n_c))
            W_xo = theano.shared(sample_weights(self.n_lstmin, n_o))
            W_ho = theano.shared(sample_weights(n_hidden, n_o))
            W_hy = theano.shared(sample_weights(n_hidden, n_y))
        
        self.W_xi = W_xi
        self.W_hi = W_hi
        self.W_xf = W_xf
        self.W_hf = W_hf
        self.W_xc = W_xc
        self.W_hc = W_hc
        self.W_xo = W_xo
        self.W_ho = W_ho
        self.W_hy = W_hy

        self.delta_W_xi = theano.shared(sample_weights(self.n_lstmin, n_i))  
        self.delta_W_hi = theano.shared(sample_weights(n_hidden, n_i))  
        self.delta_W_xf = theano.shared(sample_weights(self.n_lstmin, n_f)) 
        self.delta_W_hf = theano.shared(sample_weights(n_hidden, n_f))
        self.delta_W_xc = theano.shared(sample_weights(self.n_lstmin, n_c))  
        self.delta_W_hc = theano.shared(sample_weights(n_hidden, n_c))
        self.delta_W_xo = theano.shared(sample_weights(self.n_lstmin, n_o))
        self.delta_W_ho = theano.shared(sample_weights(n_hidden, n_o))
        self.delta_W_hy = theano.shared(sample_weights(n_hidden, n_y))

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)
        self.b_hy = theano.shared(value=numpy.zeros(n_y,dtype=dtype), name='b_hy', borrow=True)
        self.delta_b_hy = theano.shared(value=numpy.zeros(n_y,dtype=dtype), name='delta_b_hy', borrow=True)

        nCH=5
        nFrame=11
        nFrame=7
        nFeat=40
        nCHxFrame=nCH*nFrame
        self.a0_0 = theano.shared(value=numpy.ones(n_in, dtype=dtype)    / n_in, name='a0_0', borrow=True)
        self.a01_0 = theano.shared(value=numpy.ones(nFrame*nCH, dtype=dtype)    / nFrame*nCH, name='a01_0', borrow=True)
        self.a02_0 = theano.shared(value=numpy.ones(nFeat, dtype=dtype)    / nFeat, name='a02_0', borrow=True)
        self.a0_1 = theano.shared(value=numpy.ones(nFrame, dtype=dtype) / nFrame, name='a0_1', borrow=True)
        self.a0_2 = theano.shared(value=numpy.ones(nCH, dtype=dtype)    / nCH, name='a0_2', borrow=True)
        self.a0_3 = theano.shared(value=numpy.ones(nFeat, dtype=dtype)  / nFeat, name='a0_3', borrow=True)
        self.att_b1 = theano.shared(value=numpy.ones(1024,dtype=dtype) / 1024, name='att_b1', borrow=True)
        self.delta_att_b1 = theano.shared(value=numpy.ones(1024,dtype=dtype) / 1024, name='delta_att_b1', borrow=True)
        self.att_b2 = theano.shared(value=numpy.ones(nCHxFrame,dtype=dtype) / nCHxFrame, name='att_b2', borrow=True)
        self.delta_att_b2 = theano.shared(value=numpy.ones(nCHxFrame,dtype=dtype) /nCHxFrame, name='delta_att_b2', borrow=True)

        self.W0_inattend = theano.shared(value=sample_weights(n_extra_in+n_o+n_in+n_in, n_in),
                            name='W0_inattend', borrow=True)
        self.W01_inattend = theano.shared(value=sample_weights(n_extra_in+n_o+n_in+(nFrame*nCH), (nFrame*nCH)),
                            name='W01_inattend', borrow=True)
        self.W02_inattend = theano.shared(value=sample_weights(           n_o+n_in/nFrame/nCH+nFeat, nFeat),
                            name='W02_inattend', borrow=True)
        self.W1_inattend = theano.shared(value=sample_weights(n_extra_in+n_o+n_in+nFrame, nFrame),
                            name='W1_inattend', borrow=True)
        self.W2_inattend = theano.shared(value=sample_weights(n_extra_in+n_o+n_in/nFrame+nCH, nCH),
                            name='W2_inattend', borrow=True)
        self.W3_inattend = theano.shared(value=sample_weights(n_extra_in+n_o+n_in/nFrame/nCH+nFeat, nFeat),
                            name='W3_inattend', borrow=True)

        self.delta_W0_inattend = theano.shared(value = numpy.zeros((n_extra_in+n_o+n_in+n_in, n_in),
                            dtype=theano.config.floatX), name='delta_W0_inattend')
        self.delta_W01_inattend = theano.shared(value = numpy.zeros((n_extra_in+n_o+n_in+(nFrame*nCH), (nFrame*nCH)),
                            dtype=theano.config.floatX), name='delta_W01_inattend')
        self.delta_W02_inattend = theano.shared(value = numpy.zeros((           n_o+n_in/nFrame/nCH+nFeat, nFeat),
                            dtype=theano.config.floatX), name='delta_W02_inattend')
        self.delta_W1_inattend = theano.shared(value = numpy.zeros((n_extra_in+n_o+n_in+nFrame, nFrame),
                            dtype=theano.config.floatX), name='delta_W1_inattend')
        self.delta_W2_inattend = theano.shared(value = numpy.zeros((n_extra_in+n_o+n_in/nFrame+nCH, nCH),
                            dtype=theano.config.floatX), name='delta_W2_inattend')
        self.delta_W3_inattend = theano.shared(value = numpy.zeros((n_extra_in+n_o+n_in/nFrame/nCH+nFeat, nFeat),
                            dtype=theano.config.floatX), name='delta_W3_inattend')
 
        def one_lstm_step(x_t, h_tm1, c_tm1, 
                            a01_tm1,
                            W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho, 
                            W01_inattend,att_b2):
            #########################################
            #  For Attention 
            #########################################
            # 0D - ch-time-freq
            #att0_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a0_tm1, x_t)), W0_inattend))
            #att0_a_tl = T.exp(att0_e_tl)/(T.exp(att0_e_tl)).sum(0,keepdims=True)
            #att_c_t = att0_a_tl*x_t

            # 0D2 - ch-time
            e = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a01_tm1, x_t)), W01_inattend)+att_b2)
            att01_a_tl = T.exp(e)/(T.exp(e)).sum(0,keepdims=True)
            if draw != None:
                att01_a_tl = theano.printing.Print('att01_a_tl')(att01_a_tl)
            att01_c_t = T.extra_ops.repeat(att01_a_tl, 40, axis=0)*x_t # (7*5*40)*(7*5*40)
            att_c_t = att01_c_t
            # 1D - timeframe
            #att1_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a1_tm1, x_t)), W1_inattend))
            #att1_a_tl = T.exp(att1_e_tl)/(T.exp(att1_e_tl)).sum(0,keepdims=True)
            #att1_c_t = T.dot(att1_a_tl, x_t.reshape((7,5*40))).flatten() # (1,7) * ((7,5*40)) => (5*40)

            # 2D - channel
            #att2_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a2_tm1, att1_c_t)), W2_inattend))
            #att2_a_tl = T.exp(att2_e_tl)/(T.exp(att2_e_tl)).sum(0,keepdims=True)
            #att2_c_t = T.dot(att2_a_tl, att1_c_t.reshape((5,40))).flatten() # (1,5) * ((5,40)) => (1,40)

            # 3D - frequency
            #att3_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a3_tm1, att2_c_t)), W3_inattend))
            #att3_a_tl = T.exp(att3_e_tl)/(T.exp(att3_e_tl)).sum(0,keepdims=True) # 40*40
            #att_c_t = att3_a_tl*att2_c_t

            #########################################
            #  For LSTM
            #########################################
            x_t=att_c_t #rename
            i_t = T.nnet.sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi))
            f_t = T.nnet.sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf))
            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) ) 
            o_t = T.nnet.sigmoid(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho)) 
            h_t = o_t * T.tanh(c_t)
            return [h_t, c_t, att01_a_tl]
            #return [h_t, c_t, att1_a_tl,att2_a_tl,att3_a_tl]

        def one_lstm_step_wpd(x_t, extra_x_in, h_tm1, c_tm1, 
                            a01_tm1,
                            W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho, 
                            W01_inattend,att_b2):
            #########################################
            #  For Attention 
            #########################################
            # 0D - ch-time-freq
            #att0_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a0_tm1, x_t)), W0_inattend))
            #att0_a_tl = T.exp(att0_e_tl)/(T.exp(att0_e_tl)).sum(0,keepdims=True)
            #att_c_t = att0_a_tl*x_t

            # 0D2 - ch-time
            e = T.tanh(T.dot(T.join(0, extra_x_in, T.join(0, c_tm1, T.join(0, a01_tm1, x_t))), W01_inattend)+att_b2)
            att01_a_tl = T.exp(e)/(T.exp(e)).sum(0,keepdims=True)
            att01_c_t = T.extra_ops.repeat(att01_a_tl, 40, axis=0)*x_t # (7*5*40)*(7*5*40)
            att_c_t = att01_c_t
            if draw != None:
                att01_c_t = theano.printing.Print('att01_c_t')(att01_c_t)

            #e = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a02_tm1, att01_c_t)), W02_inattend))
            #att02_a_tl = T.exp(e)/(T.exp(e)).sum(0,keepdims=True) # 40*40
            #att_c_t = att02_a_tl*att01_c_t

            # 1D - timeframe
            #att1_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a1_tm1, x_t)), W1_inattend))
            #att1_a_tl = T.exp(att1_e_tl)/(T.exp(att1_e_tl)).sum(0,keepdims=True)
            #att1_c_t = T.dot(att1_a_tl, x_t.reshape((7,5*40))).flatten() # (1,7) * ((7,5*40)) => (5*40)

            # 2D - channel
            #att2_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a2_tm1, att1_c_t)), W2_inattend))
            #att2_a_tl = T.exp(att2_e_tl)/(T.exp(att2_e_tl)).sum(0,keepdims=True)
            #att2_c_t = T.dot(att2_a_tl, att1_c_t.reshape((5,40))).flatten() # (1,5) * ((5,40)) => (1,40)

            # 3D - frequency
            #att3_e_tl = T.tanh(T.dot(T.join(0, c_tm1, T.join(0, a3_tm1, att2_c_t)), W3_inattend))
            #att3_a_tl = T.exp(att3_e_tl)/(T.exp(att3_e_tl)).sum(0,keepdims=True) # 40*40
            #att_c_t = att3_a_tl*att2_c_t

            #########################################
            #  For LSTM
            #########################################
            x_t=att_c_t #rename
            i_t = T.nnet.sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi))
            f_t = T.nnet.sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf))
            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) ) 
            o_t = T.nnet.sigmoid(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho)) 
            h_t = o_t * T.tanh(c_t)
            return [h_t, c_t, att01_a_tl]

        if n_extra_in >0:
            [self.h_vals, _, _], _ = theano.scan(fn=one_lstm_step_wpd, sequences=[self.input, self.extra_x], 
                                       outputs_info=[self.h0, self.c0, self.a01_0],
					                   non_sequences=[self.W_xi, self.W_hi, 
					                   self.W_xf, self.W_hf, 
					                   self.W_xc, self.W_hc, 
					                   self.W_xo, self.W_ho,
                                       self.W01_inattend, self.att_b2],
					                   n_steps=self.n_steps, strict=True, allow_gc=False)
        else:
            [self.h_vals, _, _], _ = theano.scan(fn=one_lstm_step, sequences=self.input , 
                                       outputs_info=[self.h0, self.c0, self.a01_0],
					                   non_sequences=[self.W_xi, self.W_hi, 
					                   self.W_xf, self.W_hf, 
					                   self.W_xc, self.W_hc, 
					                   self.W_xo, self.W_ho,
                                       self.W01_inattend, self.att_b2],
					                   n_steps=self.n_steps, strict=True, allow_gc=False)
         
        self.output = T.nnet.sigmoid(theano.dot(self.h_vals,self.W_hy)+self.b_hy)
        self.y_vals = self.output

        # parameters of the model
        self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_hy, self.b_hy,
                       self.W01_inattend, self.att_b2]
        self.delta_params = [self.delta_W_xi, self.delta_W_hi, 
								self.delta_W_xf, self.delta_W_hf, 
								self.delta_W_xc, self.delta_W_hc, 
								self.delta_W_xo, self.delta_W_ho,
                                self.delta_W_hy, self.delta_b_hy,
                                self.delta_W01_inattend, self.delta_att_b2]

    def xent(self, target):
        self.cost = -T.mean(target * T.log(self.y_vals)+ (1.- target) * T.log(1. - self.y_vals))
        return self.cost

    def l2(self, target):
        error = T.mean((self.y_vals - target)**2)
        return error


