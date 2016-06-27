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
nj=10

# enhan data
enhan=beamform
enhan_data=/data-local1/nkallaku/chime3_5ch_beamform
enhan_data=/data-hpcc01/workspace/nkallaku/MAIN_DATASETS/orig_beamform/dsrtk
enhan_data=/data-local/corpora/CHiME3/data/audio/16kHz/chime3_5ch_beamform

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# process for enhan data
local/real_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;
local/simu_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc/$enhan
for x in dt05_real_$enhan tr05_real_$enhan et05_real_$enhan dt05_simu_$enhan tr05_simu_$enhan et05_simu_$enhan ; do 
  sed -i 's/-beamform//g' data/$x/spk2utt
  sed -i 's/-beamform//g' data/$x/utt2spk
  sed -i 's/-beamform_REAL/_REAL/g' data/$x/wav.scp
  sed -i 's/-beamform_SIMU/_SIMU/g' data/$x/wav.scp
  sed -i 's/-beamform_SIMU/_SIMU/g' data/$x/text
  steps/make_mfcc.sh --nj $nj \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done

