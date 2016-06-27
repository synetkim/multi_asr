#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

source_data_dir=data
target_feat=phasediff
nj=36
stage=3
list=""
list=$list" tr05_real_noisy1 dt05_real_noisy1 et05_real_noisy1"
#list=$list" tr05_real_noisy2 dt05_real_noisy2 et05_real_noisy2"
list=$list" tr05_real_noisy3 dt05_real_noisy3 et05_real_noisy3"
list=$list" tr05_real_noisy4 dt05_real_noisy4 et05_real_noisy4"
list=$list" tr05_real_noisy5 dt05_real_noisy5 et05_real_noisy5"
list=$list" tr05_real_noisy6 dt05_real_noisy6 et05_real_noisy6"
list=$list" tr05_simu_noisy1 dt05_simu_noisy1 "
#list=$list" tr05_simu_noisy2 dt05_simu_noisy2 "
list=$list" tr05_simu_noisy3 dt05_simu_noisy3 "
list=$list" tr05_simu_noisy4 dt05_simu_noisy4 "
list=$list" tr05_simu_noisy5 dt05_simu_noisy5 "
list=$list" tr05_simu_noisy6 dt05_simu_noisy6 "

target_scp_dir=data-$target_feat
target_ark_dir=$target_feat
mkdir -p $target_scp_dir
echo "=================================================="
echo "$0 $@"
echo "source: $source_data_dir "
echo "target: $target_scp_dir "
echo "   ark: $target_ark_dir "
echo "=================================================="

if [ $stage -le 2 ]; then
for x in et05_real dt05_real tr05_real dt05_simu tr05_simu; do
  rm -rf $target_scp_dir/${x}_6ch
  cp -r $source_data_dir/${x}_6ch $target_scp_dir/${x}_6ch
  steps_featext/make_${target_feat}.sh --nj $nj \
	$source_data_dir/${x}_noisy1 \
	$source_data_dir/${x}_noisy3 \
	$source_data_dir/${x}_noisy4 \
	$source_data_dir/${x}_noisy5 \
	$source_data_dir/${x}_noisy6 \
	$target_scp_dir/${x}_6ch \
	$target_ark_dir \
	exp/make_${target_feat}/$x || exit 1;
  steps/compute_cmvn_stats.sh $target_scp_dir/${x}_6ch exp/make_${target_feat}/$x $target_ark_dir || exit 1;
done
fi

enhan=6ch
utils/combine_data.sh $target_scp_dir/tr05_multi_$enhan $target_scp_dir/tr05_simu_$enhan $target_scp_dir/tr05_real_$enhan
utils/combine_data.sh $target_scp_dir/dt05_multi_$enhan $target_scp_dir/dt05_simu_$enhan $target_scp_dir/dt05_real_$enhan

echo "Done "
echo "==========================================="

