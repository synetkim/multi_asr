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


nj=36
# clean data
chime3_data=/data-local/corpora/CHiME3/
chime3_wsj1_data=/data-local/corpora/CHiME3_WSJ1/
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory
wsj1_data=$chime3_data/data/WSJ1

eval_flag=false # make it true when the evaluation data are released

# process for clean speech and making LMs etc. from original WSJ0
# note that training on clean data means original WSJ0 data only (no booth data)
#local/clean_wsj0_data_prep.sh $wsj0_data || exit 1;

#local/wsj_prepare_dict.sh || exit 1;

#utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

#local/clean_chime3_format_data.sh || exit 1; ## tr05_orig_clean dt05_orig_clean

# process for close talking speech for real data (will not be used)
#local/real_close_chime3_data_prep.sh $chime3_data || exit 1;

# process for booth recording speech (will not be used)
#local/bth_chime3_data_prep.sh $chime3_data || exit 1;

# process for distant talking speech for real and simulation data
#local/real_noisy_chime3_data_prep.sh $chime3_data || exit 1;
#local/simu_noisy_chime3_data_prep.sh $chime3_data || exit 1;
#local/simu_noisy_chime3_data_prep_wsj1.sh $chime3_wsj1_data || exit 1;

echo "========================================="
echo " Generate WSJ1 Org for using as a target(close) data"
./steps_featext/0.wsj1.close.sh  || exit 1; 
echo " Done."
echo "========================================="

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
# clean data
if $eval_flag; then
  list="tr05_orig_clean dt05_orig_clean et05_orig_clean"
else
  list="tr05_orig_clean dt05_orig_clean"
fi
# real data
if $eval_flag; then
  list=$list" tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy et05_real_close et05_real_noisy"
else
  list=$list" tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy"
fi
# simulation data
if $eval_flag; then
  list=$list" tr05_simu_noisy dt05_simu_noisy et05_simu_noisy"
else
  list=$list" tr05_simu_noisy dt05_simu_noisy"
fi

list=$list" tr05_simu_wsj1_noisy tr05_simu_wsj1_close"

mfccdir=mfcc
for x in $list; do 
  steps/make_mfcc.sh --nj $nj \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

# make mixed training set from real and simulation training data
# multi_noisy = simu + real
utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy data/tr05_simu_wsj1_noisy
utils/combine_data.sh data/dt05_multi_noisy data/dt05_simu_noisy data/dt05_real_noisy

# multi_close = simu + real
cp -rf data/tr05_orig_clean data/tr05_simu_close
cp -rf data/dt05_orig_clean data/dt05_simu_close
mv data/tr05_simu_close/spk2gender data/tr05_simu_close/spk2gender.bk
mv data/dt05_simu_close/spk2gender data/dt05_simu_close/spk2gender.bk
mv data/tr05_simu_wsj1_close/spk2gender data/tr05_simu_wsj1_close/spk2gender.bk
utils/combine_data.sh data/tr05_multi_close data/tr05_simu_close data/tr05_real_close data/tr05_simu_wsj1_close
utils/combine_data.sh data/dt05_multi_close data/dt05_simu_close data/dt05_real_close
mv data/tr05_simu_close/spk2gender.bk data/tr05_simu_close/spk2gender
mv data/dt05_simu_close/spk2gender.bk data/dt05_simu_close/spk2gender
mv data/tr05_simu_wsj1_close/spk2gender.bk data/tr05_simu_wsj1_close/spk2gender

wc -l data/tr05_multi_noisy/*
wc -l data/dt05_multi_noisy/*
wc -l data/tr05_multi_close/*
wc -l data/dt05_multi_close/*


