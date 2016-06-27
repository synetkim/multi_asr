#!/bin/bash

# Copyright 2014    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the DNN model. The features are a concatenation of the original input
# features and additional features. These additional features can be on the 
# speaker, utterance or frame level.

## Begin configuration section
stage=0
nj=16
cmd=run.pl
num_threads=1

max_active=7000 # max-active
beam=15.0 # beam used
latbeam=7.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices
max_arcs=-1

skip_scoring=false # whether to skip WER scoring
scoring_opts=

splice_opts=
norm_vars=
add_deltas=
ivec_type=speaker         # the type of the i-vectors: speaker, utterance, frame

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "Usage: steps/decode_dnn.sh [options] <graph-dir> <data-dir> <ali-dir> <iv-dir> <decode-dir>"
   echo " e.g.: steps/decode_dnn.sh exp/tri4/graph data/test exp/tri4_ali exp_ivec/ivector_eval2000 exp/tri4_dnn/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

graphdir=$1
data=$2
alidir=$3
ivdir=$4
dir=`echo $5 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
name=`basename $data`
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

[ -z "$splice_opts" ] && splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $alidir/tree $ivdir/ivector.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Generate state counts; will be used as prior
$cmd $dir/log/class_count.log \
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" ark:- \| \
    analyze-counts --binary=false ark:- $dir/class.counts || exit 1;

## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"

# Setup the additional ivector features
case $ivec_type in
  speaker) ivfeats="ark,s,cs:get-spkvec-feat --utt2spk=ark:$sdata/JOB/utt2spk scp:$ivdir/ivector.scp scp:$sdata/JOB/feats.scp ark:- |";;
  utterance) ivfeats="ark,s,cs:get-spkvec-feat scp:$ivdir/ivector.scp scp:$sdata/JOB/feats.scp ark:- |";;
  frame)  ivfeats="scp:$ivdir/ivector.scp";;
  *) echo "$0: invalid ivector type $ivec_type" && exit 1;
esac

# On each frame, append the i-vector to the original feature vector
$cmd JOB=1:$nj $dir/log/append_feat_$name.JOB.log \
  append-feats "$feats" "$ivfeats" ark,scp:$dir/append_feat.$name.JOB.ark,$dir/append_feat.$name.JOB.scp || exit 1;
##

finalfeats="ark:nnet-forward --class-frame-counts=$dir/class.counts --apply-log=true --no-softmax=false $srcdir/dnn.nnet scp:$dir/append_feat.$name.JOB.scp ark:- |"

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $alidir/final.mdl $graphdir/HCLG.fst "$finalfeats" "ark:|gzip -c > $dir/lat.JOB.gz"

rm -rf $dir/append_feat.$name.*

# Copy the source model in order for scoring
cp $alidir/final.mdl $srcdir
  
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi

exit 0;
