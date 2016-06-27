#!/bin/bash 

# Copyright 2012  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
wphase_config=conf/wphase.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: make_wphase.sh [options] <data-dir> <log-dir> <path-to-wphasedir>";
   echo "options: "
   echo "  --wphase-config <config-file>                      # config passed to compute-wphase-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
wphasedir=$3


# make $wphasedir an absolute pathname.
wphasedir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $wphasedir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $wphasedir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $wphase_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_wphase.sh: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $wphasedir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $wphasedir/raw_wphase_$name.$n.ark  
done

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_wphase_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    steps_featext/compute-wphase-feats $vtln_opts --verbose=2 --config=$wphase_config ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
     ark,scp:$wphasedir/raw_wphase_$name.JOB.ark,$wphasedir/raw_wphase_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;
 
  $cmd JOB=1:$nj $logdir/make_wphase_${name}.JOB.log \
    steps_featext/compute-wphase-feats $vtln_opts --verbose=2 --config=$wphase_config scp,p:$logdir/wav.JOB.scp ark:- \| \
    copy-feats --compress=$compress ark:- \
     ark,scp:$wphasedir/raw_wphase_$name.JOB.ark,$wphasedir/raw_wphase_$name.JOB.scp \
     || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing wphase features for $name:"
  tail $logdir/make_wphase_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $wphasedir/raw_wphase_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
