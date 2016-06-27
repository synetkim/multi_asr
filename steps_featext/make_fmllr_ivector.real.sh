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
echo "$0 $@"
if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi

nj=4

# enhan data
enhan=$1
enhan_data=$2
newdata=$2

# get alignment
steps/align_fmllr.sh --nj $nj \
  $newdata/tr05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali || exit 1;
steps/align_fmllr.sh --nj 4 \
  $newdata/dt05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali_dt05 || exit 1;

data_fmllr=data-fmllr-ivector
gmmdir=exp/tri3b_tr05_real_${enhan}
for x in dt05_real_$enhan ; do
  dir=$data_fmllr/$x
  steps/nnet/make_fmllr_ivector_feats.sh --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali_dt05 \
     $dir $newdata/$x $gmmdir $dir/log exp/nnet2_online/ivectors_dt05_real_$enhan $dir/data || exit 1
done

for x in tr05_real_$enhan ; do
  dir=$data_fmllr/$x
  steps/nnet/make_fmllr_ivector_feats.sh --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir $newdata/$x $gmmdir $dir/log exp/nnet2_online/ivectors_tr05_real_$enhan $dir/data || exit 1
done


