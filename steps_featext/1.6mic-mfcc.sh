#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

source_data_dir=data
target_feat=mfcc
nj=36
stage=2
list=""
list=$list" tr05_real_noisy1 dt05_real_noisy1 et05_real_noisy1"
list=$list" tr05_real_noisy2 dt05_real_noisy2 et05_real_noisy2"
list=$list" tr05_real_noisy3 dt05_real_noisy3 et05_real_noisy3"
list=$list" tr05_real_noisy4 dt05_real_noisy4 et05_real_noisy4"
list=$list" tr05_real_noisy5 dt05_real_noisy5 et05_real_noisy5"
list=$list" tr05_real_noisy6 dt05_real_noisy6 et05_real_noisy6"
list=$list" tr05_simu_noisy1 dt05_simu_noisy1 et05_simu_noisy1"
list=$list" tr05_simu_noisy2 dt05_simu_noisy2 et05_simu_noisy2"
list=$list" tr05_simu_noisy3 dt05_simu_noisy3 et05_simu_noisy3"
list=$list" tr05_simu_noisy4 dt05_simu_noisy4 et05_simu_noisy4"
list=$list" tr05_simu_noisy5 dt05_simu_noisy5 et05_simu_noisy5"
list=$list" tr05_simu_noisy6 dt05_simu_noisy6 et05_simu_noisy6"

target_scp_dir=data
target_ark_dir=mfcc
mkdir -p $target_scp_dir
echo "=================================================="
echo "$0 $@"
echo "source: $source_data_dir "
echo "target: $target_scp_dir "
echo "   ark: $target_ark_dir "
echo "=================================================="

if [ $stage -le 2 ]; then
for x in $list; do
#  cp -r $source_data_dir/$x $target_scp_dir
  steps/make_${target_feat}.sh --nj $nj \
    $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
  steps/compute_cmvn_stats.sh $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
done
fi

if [ $stage -le 3 ]; then
steps_featext/combine_ch.sh $target_scp_dir $target_ark_dir "$list" || exit 1;
fi

if [ $stage -le 4 ]; then
for x in tr05_real_6ch tr05_simu_6ch dt05_real_6ch dt05_simu_6ch et05_real_6ch et05_simu_6ch; do
#for x in tr05_real_6ch dt05_real_6ch et05_real_6ch ; do
  steps/compute_cmvn_stats.sh $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
done
utils/combine_data.sh $target_scp_dir/tr05_multi_6ch $target_scp_dir/tr05_simu_6ch $target_scp_dir/tr05_real_6ch
utils/combine_data.sh $target_scp_dir/dt05_multi_6ch $target_scp_dir/dt05_simu_6ch $target_scp_dir/dt05_real_6ch
fi
echo "Done "
echo "==========================================="

