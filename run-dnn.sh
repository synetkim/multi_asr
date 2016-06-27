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
testnj=`cat $working_dir/testnj`
splice_opts=`cat $working_dir/splice_opts`
split_opts=`cat $working_dir/split_opts`
norm_vars=`cat $working_dir/norm_vars`
infeat_dim=$(cat $working_dir/infeat_dim| awk '{print $2}')
outfeat_dim=$(cat $working_dir/outfeat_dim| awk '{print $2}')

echo $gpu
echo "$infeat_dim x $outfeat_dim"

#if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_DNNV.py --train-data "train.1.scp,partition=3000m,random=true,stream=true,target_dim=13,extra_dim=0" \
                                    --valid-data "valid.1.scp,partition=2000m,random=true,stream=true,target_dim=13,extra_dim=0" \
                                    --nnet-spec "1000:1024:1024:1024:$outfeat_dim" \
                                    --lrate "D:0.08:0.6:0.0001,0.0001:4" --momentum 0.9 \
                                    --l2-reg 0.2 \
                                    --batch-size 32 \
                                    --param-output-file $working_dir/dnn.param --cfg-output-file $working_dir/dnn.cfg \
                                    --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
#fi
  
echo "Generate Features"

# generate feature representations with the DNN nosoft model
for set in tr05_simu tr05_real dt05_simu dt05_real et05_simu et05_real ; do
  # set nj value
  localnj=0
  if [[ $set =~ "tr05" ]]; then
    localnj=$nj
  else
    localnj=$testnj
  fi

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

  sdata_in=$input_dir/${set}_${input_data_type}/split$localnj
  [[ -d $sdata_in && $input_dir/${set}_${input_data_type}/feats.scp -ot $sdata_in ]] || split_data.sh $split_opts $input_dir/${set}_${input_data_type} $localnj || exit 1;
  infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_in/JOB/utt2spk scp:$sdata_in/JOB/cmvn.scp scp:$sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
  
  $cmd JOB=1:$localnj $working_dir/log/make_newfeat.${set}_${input_data_type}.JOB.log \
    nnet-forward --no-softmax=true --apply-log=false $working_dir/dnn.nnet "$infeats" \
    ark,scp:$outdir/feats_predict.${set}_${input_data_type}.JOB.ark,$outdir/feats_predict.${set}_${input_data_type}.JOB.scp || exit 1;

  for n in `seq 1 $localnj`; do
    cat $outdir/feats_predict.${set}_${input_data_type}.$n.scp >> $outdir/feats.scp
  done
  cp $input_dir/${set}_${input_data_type}/spk2utt $outdir/
  cp $input_dir/${set}_${input_data_type}/utt2spk $outdir/
  cp $input_dir/${set}_${input_data_type}/text $outdir/
  steps/compute_cmvn_stats.sh $outdir/ $outdir/log $outdir/ || exit 1;
done

utils/combine_data.sh $working_dir/data-new/tr05_multi_dnnenhan $working_dir/data-new/tr05_simu_dnnenhan $working_dir/data-new/tr05_real_dnnenhan
utils/combine_data.sh $working_dir/data-new/dt05_multi_dnnenhan $working_dir/data-new/dt05_simu_dnnenhan $working_dir/data-new/dt05_real_dnnenhan

echo "Done! Generate Features"



