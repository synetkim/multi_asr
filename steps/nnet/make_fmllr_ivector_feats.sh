#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely),
#                 
# Apache 2.0.
#
# This script dumps fMLLR features in a new data directory, 
# which is later used for neural network training/testing.

# Begin configuration section.  
cmd=run.pl
transform_dir=
raw_transform_dir=
ivector_scale=1.0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 [options] <tgt-data-dir> <src-data-dir> <gmm-dir> <log-dir> <ivector-dir> <fea-dir>"
   echo "e.g.: $0 data-fmllr/train data/train exp/tri5a exp/make_fmllr_feats/log exp/ivector plp/processed/"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo "You can also use fMLLR features-- you have to supply --transform-dir option."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <transform-dir>                  # where to find fMLLR transforms."
   exit 1;
fi

data=$1
srcdata=$2
gmmdir=$3
logdir=$4
ivector_data=$5
feadir=$6

splice_opts=`cat $gmmdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $gmmdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $gmmdir/delta_opts 2>/dev/null`

mkdir -p $data $logdir $feadir

# Check files exist,
if [ ! -z "$transform_dir" ]; then
  [ ! -f $transform_dir/trans.1 ] && "$0: Missing file $transform_dir/trans.1" && exit 1;
fi

# Prepare the output dir,
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;
# Make $bnfeadir an absolute pathname,
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`

ivector_period=$(cat $ivector_data/ivector_period) || exit 1;
cat $transform_dir/trans.* > $feadir/trans

# Store the output-features,
name=`basename $data`
apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp scp:$srcdata/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | \
transform-feats $gmmdir/final.mat ark:- ark:- | transform-feats --utt2spk=ark:$data/utt2spk ark:$feadir/trans  ark:- ark:- | splice-feats --left-context=5 --right-context=5 ark:- ark:- | \
paste-feats --length-tolerance=$ivector_period ark:- "ark,s,cs:utils/filter_scp.pl $data/utt2spk $ivector_data/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- | copy-matrix --scale=$ivector_scale ark:- ark:-|" ark:- | \
 copy-feats --compress=true ark:- ark,scp:$feadir/feats_fmllr_ivector_$name.ark,$feadir/feats_fmllr_ivector_$name.scp || exit 1;

# Merge the scp,
  cat $feadir/feats_fmllr_ivector_$name.scp > $data/feats.scp

echo "$0: Done!, type $feat_type, $srcdata --> $data, using : raw-trans ${raw_transform_dir:-None}, gmm $gmmdir, trans ${transform_dir:-None}"

exit 0;
