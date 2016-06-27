#!/bin/bash

# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

nj=10

enhan=dnnenhan
data_fmllr=data-fmllr-ivector-neat

dir=exp/tri4a_dnn_tr05_multi_${enhan}_smbr_i1lats
srcdir=exp/tri4a_dnn_tr05_multi_${enhan}_smbr
acwt=0.1

for ITER in 1 2 3 4; do
  steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_multi_${enhan}/graph_tgpr_5k $data_fmllr/et05_real_${enhan} $dir/decode_tgpr_5k_et05_real_${enhan}_it${ITER} 
done
for ITER in 1 2 3 4; do
  steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_multi_${enhan}/graph_tgpr_5k $data_fmllr/et05_simu_${enhan} $dir/decode_tgpr_5k_et05_simu_${enhan}_it${ITER} 
done
