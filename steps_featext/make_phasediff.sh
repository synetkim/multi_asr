#!/bin/bash 

nj=4
cmd=run.pl
phasediff_config=conf/phasediff.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
   echo "$#"
   echo "usage: make_phasediff.sh [options] <data-dir> <data-dir2> <log-dir> <path-to-phasediff_ark_dir>";
   echo "options: "
   echo "  --phasediff-config <config-file>                      # config passed to compute-phasediff-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data1=$1
data3=$2
data4=$3
data5=$4
data6=$5
phasediff_scp_dir=$6
phasediff_ark_dir=$7
logdir=$8

# make $phasediff_ark_dir an absolute pathname.
phasediff_ark_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $phasediff_ark_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $phasediff_scp_dir`

mkdir -p $phasediff_ark_dir || exit 1;
mkdir -p $logdir || exit 1;

scp1=$data1/wav.scp
scp3=$data3/wav.scp
scp4=$data4/wav.scp
scp5=$data5/wav.scp
scp6=$data6/wav.scp

#utils/validate_data_dir.sh --no-text --no-feats $data1 || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $phasediff_ark_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $phasediff_ark_dir/raw_phasediff_$name.$n.ark  
done

if [ -f $data1/segments ]; then
  echo "$0 [info]: segments file exists: using that."
else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps1=""
  split_scps3=""
  split_scps4=""
  split_scps5=""
  split_scps6=""
  for n in $(seq $nj); do
    split_scps1="$split_scps1 $logdir/wav1.$n.scp"
    split_scps3="$split_scps3 $logdir/wav3.$n.scp"
    split_scps4="$split_scps4 $logdir/wav4.$n.scp"
    split_scps5="$split_scps5 $logdir/wav5.$n.scp"
    split_scps6="$split_scps6 $logdir/wav6.$n.scp"
  done

  utils/split_scp.pl $scp1 $split_scps1 || exit 1;
  utils/split_scp.pl $scp3 $split_scps3 || exit 1;
  utils/split_scp.pl $scp4 $split_scps4 || exit 1;
  utils/split_scp.pl $scp5 $split_scps5 || exit 1;
  utils/split_scp.pl $scp6 $split_scps6 || exit 1;

  $cmd JOB=1:$nj $logdir/make_phasediff_${name}.JOB.log \
    steps_featext/phase/compute-phasediff-feats $vtln_opts --verbose=2 --config=$phasediff_config \
			scp,p:$logdir/wav1.JOB.scp \
			scp,p:$logdir/wav3.JOB.scp \
			scp,p:$logdir/wav4.JOB.scp \
			scp,p:$logdir/wav5.JOB.scp \
			scp,p:$logdir/wav6.JOB.scp \
			ark:- \| \
    copy-feats --compress=$compress ark:- \
     ark,scp:$phasediff_ark_dir/raw_phasediff_$name.JOB.ark,$phasediff_ark_dir/raw_phasediff_$name.JOB.scp \
     || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing phasediff features for $name:"
  tail $logdir/make_phasediff_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $phasediff_ark_dir/raw_phasediff_$name.$n.scp || exit 1;
done > $phasediff_scp_dir/feats.scp
echo "Final feature scp: $phasediff_scp_dir/feats.scp"
sed -i 's/CH1//g' $phasediff_scp_dir/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $phasediff_scp_dir/feats.scp | wc -l` 
nu=`cat $phasediff_scp_dir/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
