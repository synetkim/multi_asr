#!/bin/bash

. cmd.sh


stage=1

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

enhan=$1

if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 --num-frames 200000 \
    data/tr05_multi_${enhan} 256 exp/tri3b_tr05_multi_${enhan}/ exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # use a smaller iVector dim (50) than the default (100) because RM has a very
  # small amount of data.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
    --ivector-dim 50 \
   data/tr05_multi_${enhan} exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).

  for data in tr05_multi_${enhan} dt05_multi_${enhan}; do
    utils/copy_data_dir.sh data/$data data/${data}_hires
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/${data}_hires exp/nnet2_online/extractor exp/nnet2_online/ivectors_${data} || touch exp/nnet2_online/.error 
  done

fi
