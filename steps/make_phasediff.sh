#!/bin/bash 

# Copyright 2012  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
phasediff_config=conf/phasediff.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: make_phasediff.sh [options] <data-dir> <data-dir2> <log-dir> <path-to-phasediffdir>";
   echo "options: "
   echo "  --phasediff-config <config-file>                      # config passed to compute-phasediff-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
data2=$2
logdir=$3
phasediffdir=$4


# make $phasediffdir an absolute pathname.
phasediffdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $phasediffdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $phasediffdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp
scp2=$data2/wav.scp

required="$scp $phasediff_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_phasediff.sh: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;


for n in $(seq $nj); do
  # the next command does nothing unless $phasediffdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $phasediffdir/raw_phasediff_$name.$n.ark  
done

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  split_scps2=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav1.$n.scp"
    split_scps2="$split_scps2 $logdir/wav2.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
  utils/split_scp.pl $scp2 $split_scps2 || exit 1;

  $cmd JOB=1:$nj $logdir/make_phasediff_${name}.JOB.log \
    steps_featext/compute-phasediff-feats $vtln_opts --verbose=2 --config=$phasediff_config scp,p:$logdir/wav1.JOB.scp scp,p:$logdir/wav2.JOB.scp ark:- \| \
    copy-feats --compress=$compress ark:- \
     ark,scp:$phasediffdir/raw_phasediff_$name.JOB.ark,$phasediffdir/raw_phasediff_$name.JOB.scp \
     || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing phasediff features for $name:"
  tail $logdir/make_phasediff_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $phasediffdir/raw_phasediff_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
