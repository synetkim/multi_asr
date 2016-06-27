#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely),
#                 
# Apache 2.0.
#
# This script dumps fMLLR features in a new data directory, 
# which is later used for neural network training/testing.

# Begin configuration section.  
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <tgt-data-dir> <src-data-dir> <log-dir> <ivector-dir> <fea-dir>"
   echo "e.g.: $0 data-fmllr/train data/train exp/tri5a exp/make_fmllr_feats/log exp/ivector plp/processed/"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdata=$2
logdir=$3
ivector_data=$4
feadir=$5

mkdir -p $data $logdir $feadir

# Prepare the output dir,
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;
# Make $bnfeadir an absolute pathname,
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

# Store the output-features,
name=`basename $data`

append-feats scp:$srcdata/feats.scp "scp:$ivector_data/feats.scp" ark,scp:$feadir/append_feat.$name.ark,$feadir/append_feat.$name.scp || exit 1;

# Merge the scp,
cat $feadir/append_feat.$name.scp > $data/feats.scp

echo "$0: Done!, type $feat_type, $srcdata --> $data,"

exit 0;

