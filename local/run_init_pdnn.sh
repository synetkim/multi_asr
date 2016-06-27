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

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <CHiME3 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

nj=10
newdata=exp_pdnn/data-new

#for train in tr05_multi_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
#for train in tr05_multi_noisy tr05_orig_clean; do
for train in tr05_multi_noisy ; do
  nspk=`wc -l data/$train/spk2utt | awk '{print $1}'`
  if [ $nj -gt $nspk ]; then
    nj2=$nspk
  else
    nj2=$nj
  fi
  steps/train_mono.sh --boost-silence 1.25 --nj $nj2 \
    $newdata/$train data/lang exp/mono0a_$train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj $nj2 \
    $newdata/$train data/lang exp/mono0a_$train exp/mono0a_ali_$train || exit 1;

  steps/train_deltas.sh --boost-silence 1.25 \
    2000 10000 $newdata/$train data/lang exp/mono0a_ali_$train exp/tri1_$train || exit 1;

  steps/align_si.sh --nj $nj2 \
    $newdata/$train data/lang exp/tri1_$train exp/tri1_ali_$train || exit 1;

  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 $newdata/$train data/lang exp/tri1_ali_$train exp/tri2b_$train || exit 1;

  steps/align_si.sh  --nj $nj2 \
    --use-graphs true $newdata/$train data/lang exp/tri2b_$train exp/tri2b_ali_$train  || exit 1;

  steps/train_sat.sh \
    2500 15000 $newdata/$train data/lang exp/tri2b_ali_$train exp/tri3b_$train || exit 1;

  utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_$train exp/tri3b_$train/graph_tgpr_5k || exit 1;

  # if you want to know the result of the close talk microphone, plese try the following 
  # decode close speech
  #steps/decode_fmllr.sh --nj 4 \
  #   exp/tri3b_$train/graph_tgpr_5k data/dt05_real_close exp/tri3b_$train/decode_tgpr_5k_dt05_real_close &
  # decode noisy speech
  steps/decode_fmllr.sh --nj 4 \
    exp/tri3b_$train/graph_tgpr_5k $newdata/dt05_real_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_real_noisy &
  # decode simu speech
  steps/decode_fmllr.sh --nj 4 \
    exp/tri3b_$train/graph_tgpr_5k $newdata/dt05_simu_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_simu_noisy &
done
wait

# get the best scores
#for train in tr05_multi_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
#for train in tr05_multi_noisy tr05_orig_clean; do
for train in tr05_multi_noisy ; do
  local/chime3_calc_wers.sh exp/tri3b_$train noisy \
      | tee exp/tri3b_$train/best_wer_noisy.result
done
