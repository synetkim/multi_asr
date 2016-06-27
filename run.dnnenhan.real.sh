#!/bin/bash

. ./path.sh
. ./cmd.sh 

chime3_data=/data-local/corpora/CHiME3/
enhancement_data=xx
enhancement_method=dnnenhan
local/run_gmm.newdata.real.sh $enhancement_method $enhancement_data exp_pdnn/data-new
local/run_dnn_fmllr.newdata.real.sh $enhancement_method $enhancement_data exp_pdnn/data-new

