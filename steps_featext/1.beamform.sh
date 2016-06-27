#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

source_data_dir=data
target_feat=$1
nj=36
list=""
list=$list" tr05_simu_beamform dt05_simu_beamform tr05_real_beamform dt05_real_beamform et05_real_beamform et05_simu_beamform"

target_scp_dir=data-$target_feat
target_ark_dir=$target_feat
mkdir -p $target_scp_dir
echo "=================================================="
echo "$0 $@"
echo "source: $source_data_dir "
echo "target: $target_scp_dir "
echo "   ark: $target_ark_dir "
echo "=================================================="

for x in $list ; do
  cp -r $source_data_dir/$x $target_scp_dir
  steps/make_${target_feat}.sh --nj $nj \
    $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
  steps/compute_cmvn_stats.sh $target_scp_dir/$x exp/make_${target_feat}/$x $target_ark_dir || exit 1;
done

utils/combine_data.sh $target_scp_dir/tr05_multi_beamform $target_scp_dir/tr05_simu_beamform $target_scp_dir/tr05_real_beamform
utils/combine_data.sh $target_scp_dir/dt05_multi_beamform $target_scp_dir/dt05_simu_beamform $target_scp_dir/dt05_real_beamform
echo "Done "
echo "==========================================="

