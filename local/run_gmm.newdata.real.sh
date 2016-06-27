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

if [ $# -ne 3 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

nj=4

# enhan data
enhan=$1
enhan_data=$2
newdata=$3

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# training models using enhan data
steps/train_mono.sh --boost-silence 1.25 --nj $nj \
  $newdata/tr05_real_$enhan data/lang exp/mono0a_tr05_real_$enhan || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj $nj \
  $newdata/tr05_real_$enhan data/lang exp/mono0a_tr05_real_$enhan exp/mono0a_ali_tr05_real_$enhan || exit 1;

steps/train_deltas.sh --boost-silence 1.25 \
  2000 10000 $newdata/tr05_real_$enhan data/lang exp/mono0a_ali_tr05_real_$enhan exp/tri1_tr05_real_$enhan || exit 1;

steps/align_si.sh --nj $nj \
  $newdata/tr05_real_$enhan data/lang exp/tri1_tr05_real_$enhan exp/tri1_ali_tr05_real_$enhan || exit 1;

steps/train_lda_mllt.sh \
  --splice-opts "--left-context=3 --right-context=3" \
  2500 15000 $newdata/tr05_real_$enhan data/lang exp/tri1_ali_tr05_real_$enhan exp/tri2b_tr05_real_$enhan || exit 1;

steps/align_si.sh  --nj $nj \
  --use-graphs true $newdata/tr05_real_$enhan data/lang exp/tri2b_tr05_real_$enhan exp/tri2b_ali_tr05_real_$enhan  || exit 1;

steps/train_sat.sh \
  2500 15000 $newdata/tr05_real_$enhan data/lang exp/tri2b_ali_tr05_real_$enhan exp/tri3b_tr05_real_$enhan || exit 1;

utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_$enhan/graph_tgpr_5k || exit 1;

# decode enhan speech using enhan AMs
steps/decode_fmllr.sh --nj 4 \
  exp/tri3b_tr05_real_$enhan/graph_tgpr_5k $newdata/dt05_real_$enhan exp/tri3b_tr05_real_$enhan/decode_tgpr_5k_dt05_real_$enhan &
steps/decode_fmllr.sh --nj 4 \
  exp/tri3b_tr05_real_$enhan/graph_tgpr_5k $newdata/et05_real_$enhan exp/tri3b_tr05_real_$enhan/decode_tgpr_5k_et05_real_$enhan &

wait;
# decoded results of enhan speech using enhan AMs
local/chime3_calc_wers.sh exp/tri3b_tr05_real_$enhan $enhan \
    | tee exp/tri3b_tr05_real_$enhan/best_wer_$enhan.result
