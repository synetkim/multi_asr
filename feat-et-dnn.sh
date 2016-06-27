#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

pdnndir=`pwd`/pdnn  
gpu=gpu`getUnusedGPU.sh`
working_dir=`pwd`/exp_pdnn
input_dir=`cat $working_dir/input_dir`
input_data_dir=`cat $working_dir/input_data_dir`
input_data_type=`cat $working_dir/input_data_type`
target_data_dir=`cat $working_dir/target_data_dir`
target_data_type=`cat $working_dir/target_data_type`
nj=`cat $working_dir/nj`
splice_opts=`cat $working_dir/splice_opts`
split_opts=`cat $working_dir/split_opts`
norm_vars=`cat $working_dir/norm_vars`
infeat_dim=$(cat $working_dir/infeat_dim| awk '{print $2}')
outfeat_dim=$(cat $working_dir/outfeat_dim| awk '{print $2}')

echo $gpu
echo "$infeat_dim x $outfeat_dim"

echo "Evaluation Set: Generate Features"

# generate feature representations with the DNN nosoft model
#for set in tr05_simu tr05_real dt05_simu dt05_real; do
for set in et05_real et05_simu; do
  outdir=$working_dir/data-new/${set}_dnnenhan/
  echo "inputdir = $input_dir"
  echo "outdir = $outdir"
  mkdir -p $outdir
  if [ -f $outdir/feats.scp ]; then
    rm $outdir/feats.scp
  fi
  if [ -f $outdir/cmvn.scp ]; then
    rm $outdir/cmvn.scp
  fi

  sdata_in=$input_dir/${set}_${input_data_type}/split$nj
  [[ -d $sdata_in && $input_dir/${set}_${input_data_type}/feats.scp -ot $sdata_in ]] || split_data.sh $split_opts $input_dir/${set}_${input_data_type} $nj || exit 1;
  infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_in/JOB/utt2spk scp:$sdata_in/JOB/cmvn.scp scp:$sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
  
  $cmd JOB=1:$nj $working_dir/log/make_newfeat.${set}_${input_data_type}.JOB.log \
    nnet-forward --no-softmax=true --apply-log=false $working_dir/dnn.nnet "$infeats" \
    ark,scp:$outdir/feats_predict.${set}_${input_data_type}.JOB.ark,$outdir/feats_predict.${set}_${input_data_type}.JOB.scp || exit 1;

  for n in `seq 1 $nj`; do
    cat $outdir/feats_predict.${set}_${input_data_type}.$n.scp >> $outdir/feats.scp
  done
  cp $input_dir/${set}_${input_data_type}/spk2utt $outdir/
  cp $input_dir/${set}_${input_data_type}/utt2spk $outdir/
  cp $input_dir/${set}_${input_data_type}/text $outdir/
  steps/compute_cmvn_stats.sh $outdir/ $outdir/log $outdir/ || exit 1;
done

echo "Done! Generate Features"

