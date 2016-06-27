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

import gzip
import os
import sys, re
import glob
import struct

import numpy
import theano
import theano.tensor as T
from model_io import log

class KaldiDataRead(object):

    def __init__(self, scp_path_list = [], read_opts = None):
        self.scp_path_list = scp_path_list   
        self.cur_scp_index = 0  
        self.cur_utt_index = 0  
        self.scp_file = scp_path_list[0]  
        self.read_opts = read_opts

        self.scp_file_read = None
        self.scp_cur_pos = None
        # feature information
        self.original_feat_dim = 1024
        self.feat_dim = 1024
        self.cur_frame_num = 1024
        self.target_dim = int(self.read_opts['target_dim'])
        self.extra_dim = int(self.read_opts['extra_dim'])

        self.cur_utt_num = 0
        self.max_frame_num = 0
        self.end_reading = False
        self.end_utt_reading = False

        # store features and labels for each data partition
        self.feats = numpy.zeros((10,self.feat_dim - self.target_dim), dtype=theano.config.floatX)
        if self.extra_dim > 0:
            print 'Extra feature exist'
            self.extra_feats = numpy.zeros((10, self.extra_dim), dtype=theano.config.floatX) 
        if self.target_dim == 1:
            print 'Classification'
            self.labels = numpy.zeros((10,), dtype=numpy.int32)
        else:
            print 'Regression'
            self.labels = numpy.zeros((10,self.target_dim), dtype=theano.config.floatX)

    # read the feature matrix of the next utterance
    def read_next_utt(self):
        self.scp_cur_pos = self.scp_file_read.tell()
        next_scp_line = self.scp_file_read.readline()
        if next_scp_line == '' or next_scp_line == None:    # we are reaching the end of one epoch
            return '', None
        utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')
 
        if os.path.exists(path + '.gz'):
            ark_read_buffer = gzip.open(path + '.gz', 'rb')
        else:
            ark_read_buffer = open(path, 'rb')
        ark_read_buffer.seek(int(pos),0)
       
        # now start to read the feature matrix into a numpy matrix 
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"; exit(1)

        rows = 0; cols= 0
        m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        tmp_mat = numpy.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=numpy.float32)
        utt_mat = numpy.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_id, utt_mat
 
    def is_finish(self):
        return self.end_reading

    def initialize_read(self, first_time_reading = False):
        self.scp_file_read = open(self.scp_file, 'r')
        if first_time_reading:
            utt_id, utt_mat = self.read_next_utt()
            print 'utt_id'+utt_id
            self.original_feat_dim = utt_mat.shape[1]
            self.scp_file_read = open(self.scp_file, 'r')

            # compute the feature dimension
            self.feat_dim = self.original_feat_dim - self.target_dim - self.extra_dim
            self.extra_feat_dim = self.extra_dim

            # allocate the feat matrix according to the specified partition size
            self.max_frame_num = self.read_opts['partition'] / ((self.feat_dim + self.extra_feat_dim) * 4)
            self.feats = numpy.zeros((self.max_frame_num, self.feat_dim), dtype=theano.config.floatX)
            self.extra_feats = numpy.zeros((self.max_frame_num, self.extra_feat_dim), dtype=theano.config.floatX)
            if self.target_dim == 1:
                self.labels = numpy.zeros((self.max_frame_num,), dtype=numpy.int32)
            else:
                self.labels = numpy.zeros((self.max_frame_num, self.target_dim), dtype=theano.config.floatX)
        self.end_reading = False
       
    def load_next_partition(self, shared_xye):
        self.scp_file = self.scp_path_list[self.cur_scp_index]
        if self.feats is None or len(self.scp_path_list) > 1:
		    self.scp_file_read = open(self.scp_file, 'r')

        shared_x, shared_y, shared_extra_x = shared_xye
        read_frame_num = 0
        while True:
            utt_id, utt_mat = self.read_next_utt()
            if utt_id == '':
                self.end_utt_reading = True 
                break
            rows, cols = utt_mat.shape
            if read_frame_num + rows > self.max_frame_num:
                self.scp_file_read.seek(self.scp_cur_pos)
                break
            else:
                self.feats[read_frame_num:(read_frame_num+rows)] = utt_mat[:,0:self.feat_dim]
                if self.extra_dim > 0:
                    self.extra_feats[read_frame_num:(read_frame_num+rows)] = utt_mat[:,self.feat_dim+self.target_dim:self.feat_dim+self.target_dim+self.extra_feat_dim]
                if self.target_dim == 1:
                    self.labels[read_frame_num:(read_frame_num+rows)] = utt_mat[:,self.feat_dim:self.feat_dim+self.target_dim].flatten()
                else:
                    self.labels[read_frame_num:(read_frame_num+rows)] = utt_mat[:,self.feat_dim:self.feat_dim+self.target_dim]
                read_frame_num += rows

        if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
            numpy.random.seed(18877)
            numpy.random.shuffle(self.feats[0:read_frame_num])
            numpy.random.seed(18877)
            numpy.random.shuffle(self.extra_feats[0:read_frame_num])
            numpy.random.seed(18877)
            numpy.random.shuffle(self.labels[0:read_frame_num])

        sys.stdout.flush()
 
        shared_x.set_value(self.feats[0:read_frame_num], borrow=True) 
        if self.extra_dim > 0:
            shared_extra_x.set_value(self.extra_feats[0:read_frame_num], borrow=True)
        shared_y.set_value(self.labels[0:read_frame_num], borrow=True)
        self.cur_frame_num = read_frame_num

        self.cur_scp_index += 1
        if self.cur_scp_index >= len(self.scp_path_list):
            self.end_reading = True
            self.cur_scp_index = 0

    def make_shared(self):
        shared_x = theano.shared(self.feats, name = 'x', borrow = True)
        shared_extra_x = theano.shared(self.extra_feats, name = 'extra_x', borrow = True)
        shared_y = theano.shared(self.labels, name = 'y', borrow = True)
        return shared_x, shared_y, shared_extra_x

