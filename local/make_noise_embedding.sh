#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

echo =====================================================================
dim=30
exp=exp/exp_pdnn
feature=$1
model=noise
alignment=$2
enhan=$3
echo "$feature"
echo "$dim"
echo "$model"
echo "$alignment"
echo "$enhan"
echo =====================================================================

working_dir=$exp/$model
data=$feature/
gmmdir=$alignment
gpu=gpu1
num_pdfs=4

mkdir -p $working_dir/log

echo =====================================================================
echo "                  DNN Pre-training & Fine-tuning                   "
echo =====================================================================
feat_dim=$(gunzip -c $working_dir/dt05_multi_$enhan.pfile.1.gz |head |grep num_features| awk '{print $2}') || exit 1;

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn_org/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn_org/cmds/run_DNN.py --train-data "$working_dir/tr05_multi_$enhan.pfile.*.gz,partition=2000m,random=true,stream=true" \
                                    --valid-data "$working_dir/dt05_multi_$enhan.pfile.*.gz,partition=2000m,random=true,stream=true" \
                                    --nnet-spec "$feat_dim:1024:1024:1024:$dim:1024:$num_pdfs" \
                                    --lrate "D:0.1:0.05:0.2,0.2:3" \
                                    --l2-reg 0.2 \
                                    --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
fi

echo =====================================================================
echo "                          bnf feature generation                   "
echo =====================================================================
function run() {
( cd $working_dir; ln -s dnn.nnet bnf.nnet )
set=$1
steps/compute_cmvn_stats.sh $data/$set exp/make_cmvn/$set $data/$set/data || exit 1;

if [ $set = "tr05_multi_$enhan" ];then
    steps_pdnn/make_bnf_feat.sh --nj 20 --cmd "$train_cmd" \
                $working_dir/data/$set.bnf $data/$set \
                $working_dir/ $working_dir/_log $working_dir/_bnf || exit 1
else
    steps_pdnn/make_bnf_feat.sh --nj 4 --cmd "$train_cmd" \
                $working_dir/data/$set.bnf $data/$set \
                $working_dir/ $working_dir/_log $working_dir/_bnf || exit 1
fi

steps/compute_cmvn_stats.sh \
    $working_dir/data/$set.bnf $working_dir/_log $working_dir/_bnf || exit 1;
}

run tr05_multi_$enhan
run dt05_multi_$enhan
run et05_real_$enhan
run et05_simu_$enhan

echo "Finish !!"
