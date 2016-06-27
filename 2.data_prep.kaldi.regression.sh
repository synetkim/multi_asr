#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

working_dir=`pwd`/exp_pdnn
input_dir=`pwd`/data-fbank
input_data_type=6ch
target_dir=`pwd`/data
target_data_type=close
input_data_dir=$input_dir/tr05_real_$input_data_type
target_data_dir=$target_dir/tr05_real_$target_data_type
valid_input_data_dir=$input_dir/dt05_real_$input_data_type
valid_target_data_dir=$target_dir/dt05_real_$target_data_type
nj=20
testnj=20
splice_opts="--left-context=2 --right-context=2"
split_opts="--per-utt"
norm_vars="true"

mkdir -p $working_dir/log
mkdir -p $working_dir/data

echo "$nj" > $working_dir/nj
echo "$testnj" > $working_dir/testnj
echo "$splice_opts" > $working_dir/splice_opts
echo "$split_opts" > $working_dir/split_opts
echo "$norm_vars" > $working_dir/norm_vars
echo "$input_dir" > $working_dir/input_dir
echo "$input_data_dir" > $working_dir/input_data_dir
echo "$target_data_dir" > $working_dir/target_data_dir
echo "$input_data_type" > $working_dir/input_data_type
echo "$target_data_type" > $working_dir/target_data_type

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
python steps_featext/replace_uttid.py "$input_dir" "$target_dir"

############################################
# 0.5. prepare train/cv lists
############################################
#echo "generate cv set from tr05_simu"
#if [ ! -d $target_dir/dt05_multi_new_$target_data_type ]; then
#  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $input_dir/tr05_simu_$input_data_type \
#	$input_dir/tr05_simu95_$input_data_type $input_dir/tr05_simu05_$input_data_type || exit 1
#  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $target_dir/tr05_simu_$target_data_type \
#	$target_dir/tr05_simu95_$target_data_type $target_dir/tr05_simu05_$target_data_type || exit 1
#  
#  utils/combine_data.sh $input_dir/tr05_multi_new_$input_data_type $input_dir/tr05_simu95_$input_data_type $input_dir/tr05_real_$input_data_type 
#  utils/combine_data.sh $input_dir/dt05_multi_new_$input_data_type $input_dir/tr05_simu05_$input_data_type $input_dir/dt05_real_$input_data_type 
#  utils/combine_data.sh $target_dir/tr05_multi_new_$target_data_type $target_dir/tr05_simu95_$target_data_type $target_dir/tr05_real_$target_data_type 
#  utils/combine_data.sh $target_dir/dt05_multi_new_$target_data_type $target_dir/tr05_simu05_$target_data_type $target_dir/dt05_real_$target_data_type 
#fi

############################################
# 1. split data into nj (depends on memory)
############################################
train_sdata_in=$input_data_dir/split$nj
[[ -d $train_sdata_in && $input_data_dir/feats.scp -ot $train_sdata_in ]] || split_data.sh $split_opts $input_data_dir $nj || exit 1;
train_sdata_out=$target_data_dir/split$nj
[[ -d $train_sdata_out && $target_data_dir/feats.scp -ot $train_sdata_out ]] || split_data.sh $split_opts $target_data_dir $nj || exit 1;
test_sdata_in=$valid_input_data_dir/split$testnj
[[ -d $test_sdata_in && $valid_input_data_dir/feats.scp -ot $test_sdata_in ]] || split_data.sh $split_opts $valid_input_data_dir $testnj || exit 1;
test_sdata_out=$valid_target_data_dir/split$testnj
[[ -d $test_sdata_out && $valid_target_data_dir/feats.scp -ot $test_sdata_out ]] || split_data.sh $split_opts $valid_target_data_dir $testnj || exit 1;

############################################
# 2. feature process ex) cmvn, splice
# 3. build kaldi formot => ark, scp
############################################
infeat_dim=""
outfeat_dim=""

train_infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$train_sdata_in/JOB/utt2spk scp:$train_sdata_in/JOB/cmvn.scp scp:$train_sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
test_infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$test_sdata_in/JOB/utt2spk scp:$test_sdata_in/JOB/cmvn.scp scp:$test_sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
train_outfeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$train_sdata_out/JOB/utt2spk scp:$train_sdata_out/JOB/cmvn.scp scp:$train_sdata_out/JOB/feats.scp ark:- |"
test_outfeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$test_sdata_out/JOB/utt2spk scp:$test_sdata_out/JOB/cmvn.scp scp:$test_sdata_out/JOB/feats.scp ark:- |"

$cmd JOB=1:1 $working_dir/log/get_infeat_dim.log \
	feat-to-dim "$train_infeats subset-feats --n=1 ark:- ark:- |" ark,t:$working_dir/infeat_dim || exit 1;
$cmd JOB=1:1 $working_dir/log/get_outfeat_dim.log \
	feat-to-dim "$train_outfeats subset-feats --n=1 ark:- ark:- |" ark,t:$working_dir/outfeat_dim || exit 1;
infeat_dim=`cat $working_dir/infeat_dim | awk '{print $NF}'`
outfeat_dim=`cat $working_dir/outfeat_dim | awk '{print $NF}'`

$cmd JOB=1:$nj $working_dir/log/build_train.JOB.log \
        paste-feats "$train_infeats" "$train_outfeats" ark,scp:$working_dir/data/train.JOB.ark,$working_dir/data/train.JOB.scp || exit 1;
$cmd JOB=1:$testnj $working_dir/log/build_valid.JOB.log \
        paste-feats "$test_infeats" "$test_outfeats" ark,scp:$working_dir/data/valid.JOB.ark,$working_dir/data/valid.JOB.scp || exit 1;

#$cmd JOB=2:2 $working_dir/log/build_train.pickle.JOB.log \
#        steps_pkl/build-pfile "$train_infeats" "$train_outfeats" "| python steps_pkl/build-pkl.py $working_dir/data/train.pickle.JOB.gz $infeat_dim $outfeat_dim" || exit 1;
#$cmd JOB=2:2 $working_dir/log/build_valid.pickle.JOB.log \
#        steps_pkl/build-pfile "$test_infeats" "$test_outfeats" "| python steps_pkl/build-pkl.py $working_dir/data/valid.pickle.JOB.gz $infeat_dim $outfeat_dim" || exit 1;

echo "input feature dimension: $infeat_dim"
echo "output feature dimension: $outfeat_dim"
echo "Data Processing Done."
exit 0;

