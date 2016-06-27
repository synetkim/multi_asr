#!/bin/bash

. ./path.sh
. ./cmd.sh 


# chutt.sh

chime3_data=/data-local/corpora/CHiME3/
enhancement_data=xx
enhancement_method=noisy
#local/run_gmm.newdata.real.sh $enhancement_method $enhancement_data data

#./run-cnn.sh || exit 1;

local/run_dnn_fmllr.newdata.real.sh $enhancement_method $enhancement_data data-new.fft

