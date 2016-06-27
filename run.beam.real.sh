#!/bin/bash

# Kaldi ASR baseline for the 3rd CHiME Challenge
#
# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh 

# You can execute run_init.sh only "once"
# This creates LMs, basic task files, basic models, 
# baseline results without speech enhancement techniques, and so on.
# Please set a main root directory of the CHiME3 data
# If you use kaldi scripts distributed in the CHiME3 data, 
chime3_data=/data-local/corpora/CHiME3/

# New feature representation
enhancement_method=beamform
enhancement_data=xx
local/run_gmm.newdata.real.sh $enhancement_method $enhancement_data  data
local/run_dnn.newdata.real.sh $enhancement_method $enhancement_data data

