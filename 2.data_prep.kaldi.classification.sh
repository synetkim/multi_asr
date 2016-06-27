#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

input_data_type=6ch
input_dir=`pwd`/data-fmllr
target_data_type=noisy

dataset=multi
working_dir=`pwd`/exp_pdnn.$input_data_type.$dataset
target_dir=`pwd`/data-alignment.$target_data_type.$dataset

input_data_dir=$input_dir/tr05_${dataset}_$input_data_type
target_data_dir=$target_dir/tr05_${dataset}_$target_data_type
valid_input_data_dir=$input_dir/dt05_${dataset}_$input_data_type
valid_target_data_dir=$target_dir/dt05_${dataset}_$target_data_type
nj=20
testnj=20
splice_opts="--left-context=3 --right-context=3"
split_opts="--per-utt"
norm_vars="true"
cmvn="true"
num_pdfs=`cat $target_dir/num_pdfs.$dataset`
mkdir -p $working_dir/log
mkdir -p $working_dir/data.${dataset}

echo "$nj" > $working_dir/nj
echo "$cmvn" > $working_dir/cmvn
echo "$testnj" > $working_dir/testnj
echo "$splice_opts" > $working_dir/splice_opts
echo "$split_opts" > $working_dir/split_opts
echo "$norm_vars" > $working_dir/norm_vars
echo "$input_dir" > $working_dir/input_dir
echo "$input_data_dir" > $working_dir/input_data_dir
echo "$target_data_dir" > $working_dir/target_data_dir
echo "$input_data_type" > $working_dir/input_data_type
echo "$target_data_type" > $working_dir/target_data_type
echo "$num_pdfs" > $working_dir/num_pdfs.$dataset

echo "====================================================="
echo "$0: "
echo "working_dir: $working_dir"
echo "input_data_dir: $input_data_dir"
echo "input_data_type: $input_data_type"
echo "target_data_dir: $target_data_dir"
echo "target_data_type: $target_data_type"
echo "nj:$nj testnj:$testnj splice: $splice_opts split_opts: $split_opts norm_vars: $norm_vars "
echo "====================================================="

############################################
# 0. match utterance id (source-target)
############################################
#python steps_featext/replace_uttid.py "$input_dir" "$target_dir"

############################################
# 1. split data into nj (depends on memory)
############################################
train_sdata_in=$input_data_dir/split$nj
[[ -d $train_sdata_in && $input_data_dir/feats.scp -ot $train_sdata_in ]] || split_data.sh $split_opts $input_data_dir $nj || exit 1;
echo $train_sdata_in
train_sdata_out=$target_data_dir/split$nj
[[ -d $train_sdata_out && $target_data_dir/feats.scp -ot $train_sdata_out ]] || split_data.sh $split_opts $target_data_dir $nj || exit 1;
echo $train_sdata_out
test_sdata_in=$valid_input_data_dir/split$testnj
[[ -d $test_sdata_in && $valid_input_data_dir/feats.scp -ot $test_sdata_in ]] || split_data.sh $split_opts $valid_input_data_dir $testnj || exit 1;
echo $test_sdata_in
test_sdata_out=$valid_target_data_dir/split$testnj
[[ -d $test_sdata_out && $valid_target_data_dir/feats.scp -ot $test_sdata_out ]] || split_data.sh $split_opts $valid_target_data_dir $testnj || exit 1;
echo $test_sdata_out

############################################
# 2. feature process ex) cmvn, splice
# 3. build kaldi formot => ark, scp
############################################
infeat_dim=""
outfeat_dim=""

if [ "$cmvn" == "true" ];then
echo "CMVN"
train_infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$train_sdata_in/JOB/utt2spk scp:$train_sdata_in/JOB/cmvn.scp scp:$train_sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
test_infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$test_sdata_in/JOB/utt2spk scp:$test_sdata_in/JOB/cmvn.scp scp:$test_sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
else
echo "no CMVN"
train_infeats="ark,s,cs:splice-feats $splice_opts scp:$train_sdata_in/JOB/feats.scp ark:- |"
test_infeats="ark,s,cs:splice-feats $splice_opts scp:$test_sdata_in/JOB/feats.scp ark:- |"
fi
train_outfeats="ark,s,cs:copy-feats scp:$train_sdata_out/JOB/feats.scp ark:- |"
test_outfeats="ark,s,cs:copy-feats scp:$test_sdata_out/JOB/feats.scp ark:- |"

$cmd JOB=1:1 $working_dir/log/get_infeat_dim.log \
	feat-to-dim "$train_infeats subset-feats --n=1 ark:- ark:- |" ark,t:$working_dir/infeat_dim || exit 1;
$cmd JOB=1:1 $working_dir/log/get_outfeat_dim.log \
	feat-to-dim "$train_outfeats subset-feats --n=1 ark:- ark:- |" ark,t:$working_dir/outfeat_dim || exit 1;
infeat_dim=`cat $working_dir/infeat_dim | awk '{print $NF}'`
outfeat_dim=`cat $working_dir/outfeat_dim | awk '{print $NF}'`

$cmd JOB=1:$nj $working_dir/log/build_train.JOB.log \
        paste-feats "$train_infeats" "$train_outfeats" ark,scp:$working_dir/data.$dataset/train.JOB.ark,$working_dir/data.$dataset/train.JOB.scp || exit 1;
$cmd JOB=1:$testnj $working_dir/log/build_valid.JOB.log \
        paste-feats "$test_infeats" "$test_outfeats" ark,scp:$working_dir/data.$dataset/valid.JOB.ark,$working_dir/data.$dataset/valid.JOB.scp || exit 1;

echo "input feature dimension: $infeat_dim"
echo "output feature dimension: $outfeat_dim"
echo "Data Processing Done."
exit 0;

