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
#local/run_init_short.sh $chime3_data

# GMM based ASR experiment
# Please set a directory of your speech enhancement method.
# run_gmm.sh can be done every time when you change a speech enhancement technique.
# The directory structure and audio files must follow the attached baseline enhancement directory

enhancement_method=beamform
enhancement_data=/data-local1/nkallaku/chime3_5ch_beamform
#local/run_gmm.real.beam.sh $enhancement_method xx || exit 1

# DNN based ASR experiment
# Since it takes time to evaluate DNN, we make the GMM and DNN scripts separately.
# You may execute it after you would have promising results using GMM-based ASR experiments

DNN_TYPE=ivector

# FMLLR + IVECTOR
if [ "$DNN_TYPE" = "ivector" ]; then
  steps_featext/make_ivector_hires.sh $enhancement_method 
  local/make_fmllr_ivector.sh $enhancement_method $enhancement_data || exit 1
#  local/run_dnn_fmllr_ivector.sh $enhancement_method $enhancement_data || exit 1
fi
