#!/bin/bash

. cmd.sh

nj=12 #MULTI
nj=4 #REAL 
stage=1
mfccdir=mfcc

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# enhan data
enhan=$1


if [ $stage -le 1 ]; then
  for datadir in tr05_real_${enhan} dt05_real_${enhan} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
  done
fi


if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align the si84 data for this purpose.

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/tr05_real_${enhan} data/lang exp/tri3b_tr05_real_${enhan} exp/nnet2_online/tri3b_tr05_real_${enhan}_ali || exit 1;
fi


if [ $stage -le 3 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/tr05_real_${enhan}_hires data/lang \
     exp/nnet2_online/tri3b_tr05_real_${enhan}_ali exp/nnet2_online/tri4b_tr05_real_${enhan} || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
     --num-frames 400000 data/tr05_real_${enhan}_hires 256 exp/nnet2_online/tri4b_tr05_real_${enhan} exp/nnet2_online/diag_ubm || exit 1;
fi

if [ $stage -le 5 ]; then
  # even though $nj is just 10, each job uses realple processes and threads.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj  --ivector-dim 100 \
    data/tr05_real_${enhan}_hires exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  # We extract iVectors on all the train_si284 data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).

#  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/tr05_real_${enhan}_hires \
#    data/tr05_real_${enhan}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/tr05_real_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_tr05_real_${enhan} || exit 1;
fi

if [ $stage -le 7 ]; then
  rm exp/nnet2_online/.error 2>/dev/null
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/dt05_real_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_dt05_real_${enhan} || touch exp/nnet2_online/.error &
  wait
  [ -f exp/nnet2_online/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

