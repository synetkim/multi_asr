#!/bin/bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 29/05/12

# Modified from the script for CHiME3 baseline
# Shinji Watanabe 02/13/2015

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level CHiME3_WSJ1 directory."
  echo "It is assumed that there will be a 'data' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

eval_flag=false # make it true when the evaluation data are released

audio_dir=$1/data/audio/16kHz/isolated
trans_dir=$1/data/transcriptions

echo "extract 5th channel (CH5.wav, the center bottom edge in the front of the tablet) for noisy data"

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

list_set="tr05_simu_wsj1_noisy"

cd $dir

find $audio_dir -name '*CH5.wav' | grep 'tr05_bus_simu_new\|tr05_caf_simu_new\|tr05_ped_simu_new\|tr05_str_simu_new' | sort -u > tr05_simu_wsj1_noisy.flist

# make a dot format from json annotation files
find $trans_dir -name "*.dot" >  dot_files_wsj1.flist

# make a scp file from file list
for x in $list_set; do
    cat $x.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.wav/_SIMU/' > ${x}_wav.ids
    paste -d" " ${x}_wav.ids $x.flist | sort -k 1 > ${x}_wav.scp
done

# make a transcription from dot
# simulation training data extract dot file from original WSJ1 data
# since it is generated from these data
cat tr05_simu_wsj1_noisy_wav.scp | awk -F'[_]' '{print $2}' | tr '[A-Z]' '[a-z]' \
    | $local/find_noisy_transcripts.pl dot_files_wsj1.flist | cut -f 2- -d" " > tr05_simu_wsj1_noisy.txt
cat tr05_simu_wsj1_noisy_wav.scp | cut -f 1 -d" " > tr05_simu_wsj1_noisy.ids
paste -d" " tr05_simu_wsj1_noisy.ids tr05_simu_wsj1_noisy.txt | sort -k 1 > tr05_simu_wsj1_noisy.trans1


# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in $list_set;do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done
 
# Make the utt2spk and spk2utt files.
for x in $list_set; do
  cat ${x}_wav.scp | awk -F'_' '{print $1}' > $x.spk
  cat ${x}_wav.scp | awk '{print $1}' > $x.utt
  paste -d" " $x.utt $x.spk > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# copying data to data/...
for x in $list_set; do
  mkdir -p ../../$x
  cp ${x}_wav.scp ../../$x/wav.scp || exit 1;
  cp ${x}.txt     ../../$x/text    || exit 1;
  cp ${x}.spk2utt ../../$x/spk2utt || exit 1;
  cp ${x}.utt2spk ../../$x/utt2spk || exit 1;
done

echo "Data preparation succeeded"
