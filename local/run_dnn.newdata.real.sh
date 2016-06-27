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

if [ $# -ne 3 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi


# enhan data
enhan=$1
enhan_data=$2
newdata=$3
nj=4 #should match exp/tri3b nj

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_real_$enhan ]; then
  echo "error, execute local/run_gmm.sh, first"
  exit 1;
fi

steps/align_fmllr.sh --nj $nj \
  data/tr05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali || exit 1;
steps/align_fmllr.sh --nj 4 \
  data/dt05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali_dt05 || exit 1;

data_fmllr=$newdata
gmmdir=exp/tri3b_tr05_real_${enhan}

# pre-train dnn
dir=exp/tri4a_dnn_pretrain_tr05_real_$enhan
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 $data_fmllr/tr05_real_$enhan $dir

# train dnn
dir=exp/tri4a_dnn_tr05_real_$enhan
ali=exp/tri3b_tr05_real_${enhan}_ali
ali_dev=exp/tri3b_tr05_real_${enhan}_ali_dt05 
feature_transform=exp/tri4a_dnn_pretrain_tr05_real_$enhan/final.feature_transform
dbn=exp/tri4a_dnn_pretrain_tr05_real_$enhan/7.dbn
$cuda_cmd $dir/_train_nnet.log \
steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
$data_fmllr/tr05_real_$enhan $data_fmllr/dt05_real_$enhan data/lang $ali $ali_dev $dir || exit 1;

# decode enhan speech
utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config \
  $dir/graph_tgpr_5k $data_fmllr/dt05_real_$enhan $dir/decode_tgpr_5k_dt05_real_$enhan &
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config \
  $dir/graph_tgpr_5k $data_fmllr/et05_real_$enhan $dir/decode_tgpr_5k_et05_real_$enhan &
wait;

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/tri4a_dnn_tr05_real_${enhan}_smbr
srcdir=exp/tri4a_dnn_tr05_real_${enhan}
acwt=0.1

# First we generate lattices and alignments:
# gawk musb be installed to perform awk -v FS="/" '{ print gensub(".gz","","",$NF)" gunzip -c "$0" |"; }' in
# steps/nnet/make_denlats.sh
steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali
steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_denlats

# Re-train the DNN by 1 iteration of sMBR
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir

# Decode (reuse HCLG graph)
#for ITER in 1; do
#  steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
#    --nnet $dir/${ITER}.nnet --acwt $acwt \
#    exp/tri4a_dnn_tr05_real_${enhan}/graph_tgpr_5k $data_fmllr/dt05_real_${enhan} $dir/decode_tgpr_5k_dt05_real_${enhan}_it${ITER} &
#done

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/tri4a_dnn_tr05_real_${enhan}_smbr_i1lats
srcdir=exp/tri4a_dnn_tr05_real_${enhan}_smbr
acwt=0.1

# Generate lattices and alignments:
steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali
steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_denlats

# Re-train the DNN by 4 iterations of sMBR
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
  $data_fmllr/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1

# Decode (reuse HCLG graph)
for ITER in 1 2 3 4; do
  steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_real_${enhan}/graph_tgpr_5k $data_fmllr/dt05_real_${enhan} $dir/decode_tgpr_5k_dt05_real_${enhan}_it${ITER} &
  steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_real_${enhan}/graph_tgpr_5k $data_fmllr/et05_real_${enhan} $dir/decode_tgpr_5k_et05_real_${enhan}_it${ITER} &
wait
done

# decoded results of enhan speech using enhan DNN AMs
local/chime3_calc_wers.sh exp/tri4a_dnn_tr05_real_$enhan $enhan \
    | tee exp/tri4a_dnn_tr05_real_$enhan/best_wer_$enhan.result
# decoded results of enhan speech using enhan DNN AMs with sequence training
./local/chime3_calc_wers_smbr.sh exp/tri4a_dnn_tr05_real_${enhan}_smbr_i1lats ${enhan} exp/tri4a_dnn_tr05_real_${enhan}/graph_tgpr_5k \
    | tee exp/tri4a_dnn_tr05_real_${enhan}_smbr_i1lats/best_wer_${enhan}.result

