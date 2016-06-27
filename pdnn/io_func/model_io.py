# Copyright 2013    Yajie Miao    Carnegie Mellon University 

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

# Various functions to write models from nets to files, and to read models from
# files to nets

import numpy as np
import os
import sys
import cPickle

from StringIO import StringIO
import json

import theano
import theano.tensor as T

from datetime import datetime

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

# convert an array to a string
def array_2_string(array):
    str_out = StringIO()
    np.savetxt(str_out, array)
    return str_out.getvalue()

# convert a string to an array
def string_2_array(string):
    str_in = StringIO(string)
    return np.loadtxt(str_in)

def _nnet2file(layers, extra_layers = None, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
       set_layer_num = n_layers

    for i in range(start_layer, set_layer_num):
       layer = layers[i]
       dict_a = 'W' + str(i)
       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       if layer.type == 'fc':
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W.get_value())
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b.get_value())
       elif layer.type == 'conv':
           filter_shape = layer.filter_shape
           for next_X in xrange(filter_shape[0]):
               for this_X in xrange(filter_shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string((1.0-dropout_factor) * (layer.W.get_value())[next_X, this_X])
           dict_a = 'b' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b.get_value())
       elif layer.type == 'lstmv':
           dict_a = 'W_ci' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.W_ci.get_value())
           dict_a = 'W_cf' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.W_cf.get_value())
           dict_a = 'W_co' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.W_co.get_value())
           dict_a = 'b_i' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_i.get_value())
           dict_a = 'b_f' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_f.get_value())
           dict_a = 'b_o' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_o.get_value())
           dict_a = 'b_c' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_c.get_value())
           dict_a = 'b_hy' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_hy.get_value())
           dict_a = 'W_hy' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hy.get_value())
           dict_a = 'W_xi' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xi.get_value())
           dict_a = 'W_hi' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hi.get_value())
           dict_a = 'W_xf' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xf.get_value())
           dict_a = 'W_hf' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hf.get_value())
           dict_a = 'W_xc' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xc.get_value())
           dict_a = 'W_hc' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hc.get_value())
           dict_a = 'W_xo' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xo.get_value())
           dict_a = 'W_ho' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_ho.get_value())
       elif layer.type == 'attendlstm':
           dict_a = 'att_b1' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.att_b1.get_value())
           dict_a = 'att_b2' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.att_b2.get_value())
           dict_a = 'W0_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W0_inattend.get_value())
           dict_a = 'W01_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W01_inattend.get_value())
           dict_a = 'W02_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W02_inattend.get_value())
           dict_a = 'W1_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W1_inattend.get_value())
           dict_a = 'W2_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W2_inattend.get_value())
           dict_a = 'W3_inattend' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W3_inattend.get_value())
           dict_a = 'b_hy' + str(i)
           nnet_dict[dict_a] = array_2_string(layer.b_hy.get_value())
           dict_a = 'W_hy' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hy.get_value())
           dict_a = 'W_xi' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xi.get_value())
           dict_a = 'W_hi' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hi.get_value())
           dict_a = 'W_xf' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xf.get_value())
           dict_a = 'W_hf' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hf.get_value())
           dict_a = 'W_xc' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xc.get_value())
           dict_a = 'W_hc' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_hc.get_value())
           dict_a = 'W_xo' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_xo.get_value())
           dict_a = 'W_ho' + str(i)
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * layer.W_ho.get_value())

    with open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush() 


# save the config classes; since we are using pickle to serialize the whole class, it's better to set the
# data reading and learning rate interfaces to None.
def _cfg2file(cfg, filename='cfg.out'):
    cfg.lrate = None
    cfg.train_sets = None; cfg.train_xy = None; cfg.train_x = None; cfg.train_y = None
    cfg.valid_sets = None; cfg.valid_xy = None; cfg.valid_x = None; cfg.valid_y = None
    cfg.train_xye = None;  cfg.extra_train_x = None; cfg.extra_train_sets = None;
    cfg.valid_xye = None;  cfg.extra_valid_x = None; cfg.extra_valid_sets = None;
    cfg.activation = None  # saving the rectifier function causes errors; thus we don't save the activation function
                           # the activation function is initialized from the activation text ("sigmoid") when the network
                           # configuration is loaded
    with open(filename, "wb") as output:
        cPickle.dump(cfg, output, cPickle.HIGHEST_PROTOCOL)

def _file2nnet(layers, extra_layers = None, set_layer_num = -1, filename='nnet.in', start_layer = 0, factor=1.0):
    print '[_file2nnet] from: '+str(start_layer)+' - until: '+str(set_layer_num) +' ==> Total layer #: '+str(len(layers))
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
        set_layer_num = n_layers

    with open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)

    if extra_layers == None:
       print "\tno extra layers"
    else:
       for i in range(0, len(extra_layers)):
           print "extra layer:" + str(i)
           extra_layer = extra_layers[i]
           dict_a = 'extra_W' + str(i)
           extra_layer.W.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))  
           dict_a = 'extra_b' + str(i)
           extra_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))

    for i in range(start_layer, set_layer_num):
        print '============= Load Param Layer: ' +str(i) + ' ============= '
        dict_a = 'W' + str(i)
        layer = layers[i-start_layer]
        if layer.type == 'fc':
            layer.W.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b' + str(i) 
            layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
        elif layer.type == 'conv':
            print 'conv'+str(i) 
            filter_shape = layer.filter_shape
            W_array = layer.W.get_value()
            for next_X in xrange(filter_shape[0]):
                for this_X in xrange(filter_shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX)
            layer.W.set_value(W_array)
            dict_a = 'b' + str(i) 
            layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
        elif layer.type == 'lstmv':
            dict_a = 'W_ci' + str(i)
            layer.W_ci.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_cf' + str(i)
            layer.W_cf.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_co' + str(i)
            layer.W_co.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_i' + str(i)
            layer.b_i.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_f' + str(i)
            layer.b_f.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_c' + str(i)
            layer.b_c.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_o' + str(i)
            layer.b_o.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_hy' + str(i)
            layer.b_hy.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hy' + str(i)
            layer.W_hy.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xi' + str(i)
            layer.W_xi.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hi' + str(i)
            layer.W_hi.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xf' + str(i)
            layer.W_xf.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hf' + str(i)
            layer.W_hf.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xc' + str(i)
            layer.W_xc.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hc' + str(i)
            layer.W_hc.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xo' + str(i)
            layer.W_xo.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_ho' + str(i)
            layer.W_ho.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
        elif layer.type == 'attendlstm':
            dict_a = 'att_b1' + str(i)
            layer.att_b1.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'att_b2' + str(i)
            layer.att_b2.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W0_inattend' + str(i)
            layer.W0_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W01_inattend' + str(i)
            layer.W01_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W02_inattend' + str(i)
            layer.W02_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W1_inattend' + str(i)
            layer.W1_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W2_inattend' + str(i)
            layer.W2_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W3_inattend' + str(i)
            layer.W3_inattend.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'b_hy' + str(i)
            layer.b_hy.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hy' + str(i)
            layer.W_hy.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xi' + str(i)
            layer.W_xi.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hi' + str(i)
            layer.W_hi.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xf' + str(i)
            layer.W_xf.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hf' + str(i)
            layer.W_hf.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xc' + str(i)
            layer.W_xc.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_hc' + str(i)
            layer.W_hc.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_xo' + str(i)
            layer.W_xo.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))
            dict_a = 'W_ho' + str(i)
            layer.W_ho.set_value(factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
            print str(layer.type)+str(i)+': '+str(dict_a)+str(np.asarray(string_2_array(nnet_dict[dict_a]).shape))

    sys.stdout.flush() 

def _cnn2file(conv_layers, filename='nnet.out', input_factor = 1.0, factor=[]):
    n_layers = len(conv_layers)
    nnet_dict = {}
    for i in xrange(n_layers):
       conv_layer = conv_layers[i]
       filter_shape = conv_layer.filter_shape
       
       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       for next_X in xrange(filter_shape[0]):
           for this_X in xrange(filter_shape[1]):
               dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X) 
               nnet_dict[dict_a] = array_2_string(dropout_factor * (conv_layer.W.get_value())[next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())
    
    with open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _file2cnn(conv_layers, filename='nnet.in', factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}

    with open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(n_layers):
        conv_layer = conv_layers[i]
        filter_shape = conv_layer.filter_shape
        W_array = conv_layer.W.get_value()

        for next_X in xrange(filter_shape[0]):
            for this_X in xrange(filter_shape[1]):
                dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
                W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

        conv_layer.W.set_value(W_array) 

        dict_a = 'b' + str(i)
        conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)) 
