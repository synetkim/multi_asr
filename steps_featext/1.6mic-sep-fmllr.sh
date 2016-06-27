#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

source_data_dir=data
target_feat=fmllr
nj=4
stage=0

dataset=multi
target_scp_dir=data-$target_feat
target_ark_dir=$target_feat
mkdir -p $target_scp_dir
echo "=================================================="
echo "$0 $@"
echo "dataset: $dataset "
echo " source: $source_data_dir "
echo " target: $target_scp_dir "
echo "    ark: $target_ark_dir "
echo "=================================================="

list="noisy1 noisy3 noisy4 noisy5 noisy6"
list="noisy6 "

for enhan in $list ; do

  tri3dir=exp/tri3b_tr05_${dataset}_${enhan}

  # get alignments
  if [ $stage -le 0 ]; then
    utils/combine_data.sh data/tr05_multi_$enhan data/tr05_simu_$enhan data/tr05_real_$enhan
    utils/combine_data.sh data/dt05_multi_$enhan data/dt05_simu_$enhan data/dt05_real_$enhan

    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/tr05_${dataset}_$enhan data/lang $tri3dir exp/tri3b_tr05_${dataset}_${enhan}_ali
    steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
      data/dt05_${dataset}_$enhan data/lang $tri3dir exp/tri3b_tr05_${dataset}_${enhan}_ali_dt05
  fi

  # make fmllr feature for training multi = simu + real
  gmmdir=exp/tri3b_tr05_${dataset}_${enhan}_ali
  data_fmllr=$target_scp_dir
  mkdir -p $data_fmllr
  fmllrdir=$target_ark_dir/$enhan
  if [ $stage -le 1 ]; then
    for x in tr05_real_$enhan tr05_simu_$enhan; do
      steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" \
        --transform-dir $gmmdir \
        $data_fmllr/$x data/$x $gmmdir exp/make_fmllr_tri3b/$x $fmllrdir
      steps/compute_cmvn_stats.sh $data_fmllr/$x exp/make_fmllr_tri3b/$x $fmllrdir || exit 1;
    done
  fi

  # make fmllr feature for dev and eval
  if [ $stage -le 2 ]; then
    for x in dt05_real_$enhan et05_real_$enhan dt05_simu_$enhan ; do
      steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" \
        --transform-dir $tri3dir/decode_tgpr_5k_${x} \
        $data_fmllr/$x data/$x $tri3dir exp/make_fmllr_tri3b/$x $fmllrdir
      steps/compute_cmvn_stats.sh $data_fmllr/$x exp/make_fmllr_tri3b/$x $fmllrdir || exit 1;
    done
  fi

  # make mixed training set from real and simulation enhanced data
  # multi = simu + real
  if [ $stage -le 3 ]; then
    utils/combine_data.sh $data_fmllr/tr05_multi_$enhan $data_fmllr/tr05_simu_$enhan $data_fmllr/tr05_real_$enhan
    utils/combine_data.sh $data_fmllr/dt05_multi_$enhan $data_fmllr/dt05_simu_$enhan $data_fmllr/dt05_real_$enhan
  fi

done

if [ $stage -le 4 ]; then
	steps_featext/combine_ch.sh data-fmllr fmllr "$list" || exit 1;
fi
if [ $stage -le 5 ]; then
	for x in tr05_real_6ch tr05_simu_6ch dt05_real_6ch dt05_simu_6ch et05_real_6ch ; do
  		steps/compute_cmvn_stats.sh $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
	done
fi
utils/combine_data.sh data-fmllr/tr05_multi_6ch data-fmllr/tr05_simu_6ch data-fmllr/tr05_real_6ch
utils/combine_data.sh data-fmllr/dt05_multi_6ch data-fmllr/dt05_simu_6ch data-fmllr/dt05_real_6ch

echo "Done "
echo "==========================================="

