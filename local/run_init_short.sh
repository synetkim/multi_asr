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

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

nj=10
# clean data
chime3_data=$1
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory

eval_flag=false # make it true when the evaluation data are released

# process for clean speech and making LMs etc. from original WSJ0
# note that training on clean data means original WSJ0 data only (no booth data)
local/clean_wsj0_data_prep.sh $wsj0_data || exit 1;

local/wsj_prepare_dict.sh || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/clean_chime3_format_data.sh || exit 1; ## tr05_orig_clean dt05_orig_clean

# process for close talking speech for real data (will not be used)
local/real_close_chime3_data_prep.sh $chime3_data || exit 1;

# process for distant talking speech for real and simulation data
local/real_noisy_chime3_data_prep.sh $chime3_data || exit 1;
local/simu_noisy_chime3_data_prep.sh $chime3_data || exit 1;

local/real_close_chime3_data_prep_eval.sh $chime3_data || exit 1;
local/real_noisy_chime3_data_prep_eval.sh $chime3_data || exit 1;
local/simu_noisy_chime3_data_prep_eval.sh $chime3_data || exit 1;


