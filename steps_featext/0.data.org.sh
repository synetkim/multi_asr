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
chime3_data=/data-local/corpora/CHiME3/
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory
#wsj0_data=/data-hpcc01/workspace/nkallaku/MAIN_DATASETS/orig_beamform # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory
stage=3

if [ $stage -le 2 ]; then
local/clean_wsj0_data_prep.sh $wsj0_data || exit 1;
local/wsj_prepare_dict.sh || exit 1;
utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
fi

# process for distant talking speech for real and simulation data
local/clean_chime3_format_data.sh || exit 1; ## tr05_orig_clean dt05_orig_clean
local/real_close_chime3_data_prep.sh $chime3_data  || exit 1;  
local/real_noisy_chime3_data_prep.sh $chime3_data  || exit 1;
local/simu_noisy_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch6_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch6_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch5_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch5_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch4_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch4_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch3_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch3_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch2_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch2_chime3_data_prep.sh $chime3_data || exit 1;
local/real_ch1_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_ch1_chime3_data_prep.sh $chime3_data || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 1 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 1 || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 2 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 2 || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 3 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 3 || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 4 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 4 || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 5 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 5 || exit 1;
local/real_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 6 || exit 1;
local/simu_noisy_chime3_data_prep_eval.6ch.sh $chime3_data 6 || exit 1;

mv data/tr05_orig_clean data/tr05_simu_close
mv data/dt05_orig_clean data/dt05_simu_close
mv data/et05_orig_clean data/et05_simu_close


