#!/bin/bash

. cmd.sh

nj=12
stage=1
sf2_matdir=sf2_mat

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# enhan data
enhan=$1


if [ $stage -le 1 ]; then
  echo "=============================1"
  for datadir in et05_real_${enhan} dt05_real_${enhan} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_sf2_mat_etdt.real.sh --nj $nj --sf2-config conf/sf2_mat_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
  done
  for datadir in et05_simu_${enhan} dt05_simu_${enhan} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_sf2_mat_etdt.simu.sh --nj $nj --sf2-config conf/sf2_mat_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
  done
  for datadir in tr05_real_${enhan} tr05_simu_${enhan} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_sf2_mat.sh --nj $nj --sf2-config conf/sf2_mat_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $sf2_matdir || exit 1;
  done
  utils/combine_data.sh data/tr05_multi_${enhan}_hires data/tr05_simu_${enhan}_hires data/tr05_real_${enhan}_hires
  utils/combine_data.sh data/dt05_multi_${enhan}_hires data/dt05_simu_${enhan}_hires data/dt05_real_${enhan}_hires
fi


if [ $stage -le 2 ]; then
  echo "=============================2"
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align the si84 data for this purpose.

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/tr05_multi_${enhan} data/lang exp/tri3b_tr05_multi_${enhan} exp/nnet2_online/tri3b_tr05_multi_${enhan}_ali
fi


if [ $stage -le 3 ]; then
  echo "=============================3"
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/tr05_multi_${enhan}_hires data/lang \
     exp/nnet2_online/tri3b_tr05_multi_${enhan}_ali exp/nnet2_online/tri4b_tr05_multi_${enhan}
fi

if [ $stage -le 4 ]; then
  echo "=============================4"
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
     --num-frames 400000 data/tr05_multi_${enhan}_hires 2048 exp/nnet2_online/tri4b_tr05_multi_${enhan} exp/nnet2_online/diag_ubm
fi

if [ $stage -le 5 ]; then
  echo "=============================5"
  # even though $nj is just 10, each job uses multiple processes and threads.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj  --ivector-dim 100 \
    data/tr05_multi_${enhan}_hires exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "=============================6"
  # We extract iVectors on all the train_si284 data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).

#  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/tr05_multi_${enhan}_hires \
#    data/tr05_multi_${enhan}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/tr05_multi_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_tr05_multi_${enhan} || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "=============================7"
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/dt05_multi_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_dt05_multi_${enhan} || exit 1;

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/et05_real_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_et05_real_${enhan} || touch exp/nnet2_online/.error &
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/et05_simu_${enhan}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_et05_simu_${enhan} || touch exp/nnet2_online/.error &

fi

echo "=============================Done"
