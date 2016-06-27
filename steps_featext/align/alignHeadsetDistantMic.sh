#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

org_data_dir=/data-local1/sykim/mend.fordata/data
aligned_scp_dir=data-align
aligned_ark_dir=aligned_ark_dir

echo "==========================================="
echo "$0 $@"
echo "org_data_dir: $org_data_dir"
echo "aligned_scp_dir: $aligned_scp_dir"
echo "aligned_ark_dir: $aligned_ark_dir"
echo "==========================================="

#for set in tr05_real tr05_simu dt05_real dt05_simu et05_real et05_simu; do
for set in tr05_real tr05_simu dt05_real et05_real ; do
echo "Start set: $set"
orgh=$org_data_dir/${set}_close/wav.scp
orgc1=$org_data_dir/${set}_noisy1/wav.scp
orgc3=$org_data_dir/${set}_noisy3/wav.scp
orgc4=$org_data_dir/${set}_noisy4/wav.scp
orgc5=$org_data_dir/${set}_noisy5/wav.scp
orgc6=$org_data_dir/${set}_noisy6/wav.scp
mkdir -p $aligned_scp_dir/${set}_close
mkdir -p $aligned_scp_dir/${set}_noisy1
mkdir -p $aligned_scp_dir/${set}_noisy3
mkdir -p $aligned_scp_dir/${set}_noisy4
mkdir -p $aligned_scp_dir/${set}_noisy5
mkdir -p $aligned_scp_dir/${set}_noisy6
mkdir -p $aligned_ark_dir/${set}_close
mkdir -p $aligned_ark_dir/${set}_noisy1
mkdir -p $aligned_ark_dir/${set}_noisy3
mkdir -p $aligned_ark_dir/${set}_noisy4
mkdir -p $aligned_ark_dir/${set}_noisy5
mkdir -p $aligned_ark_dir/${set}_noisy6
newh=$aligned_scp_dir/${set}_close/wav.scp
newc1=$aligned_scp_dir/${set}_noisy1/wav.scp
newc3=$aligned_scp_dir/${set}_noisy3/wav.scp
newc4=$aligned_scp_dir/${set}_noisy4/wav.scp
newc5=$aligned_scp_dir/${set}_noisy5/wav.scp
newc6=$aligned_scp_dir/${set}_noisy6/wav.scp
arkh=$aligned_ark_dir/${set}_close/wav.ark
arkc1=$aligned_ark_dir/${set}_noisy1/wav.ark
arkc3=$aligned_ark_dir/${set}_noisy3/wav.ark
arkc4=$aligned_ark_dir/${set}_noisy4/wav.ark
arkc5=$aligned_ark_dir/${set}_noisy5/wav.ark
arkc6=$aligned_ark_dir/${set}_noisy6/wav.ark

./steps_featext/align/align-wav scp:$orgh scp:$orgc5 scp:$orgc1 scp:$orgc3 scp:$orgc4 scp:$orgc6 \
	ark,scp:$arkh,$newh \
	ark,scp:$arkc5,$newc5 \
	ark,scp:$arkc1,$newc1 \
	ark,scp:$arkc3,$newc3 \
	ark,scp:$arkc4,$newc4 \
	ark,scp:$arkc6,$newc6   &
sleep 1;
done
wait;
echo "Done "
echo "==========================================="
