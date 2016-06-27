#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

if [ ! -d local/pfile_utils-v0_51/bin ]; then
  cd local/
  ./install_pfile_utils.sh
  cd -
fi

echo =====================================================================
exp=exp/exp_pdnn
feature=$1
model=noise
alignment=$2
enhan=$3
echo $feature
echo $model
echo $alignment
echo $enhan
echo =====================================================================

working_dir=$exp/$model/
data=$feature/
gmmdir=$alignment
gpu=gpu1
mkdir -p $working_dir

steps/compute_cmvn_stats.sh $data/tr05_multi_$enhan exp/make_cmvn/tr05_multi_$enhan $data/data || exit 1;
steps/compute_cmvn_stats.sh $data/dt05_multi_$enhan exp/make_cmvn/dt05_multi_$enhan $data/data || exit 1;

function run() {

steps_pdnn/build_nnet_pfile_noise_auto.sh --cmd "$train_cmd" --norm-vars false --do-concat false \
      --splice-opts "--left-context=5 --right-context=5" \
      $data/tr05_multi_$enhan ${gmmdir}_ali $working_dir || exit 1
steps_pdnn/build_nnet_pfile_noise_auto.sh --cmd "$train_cmd" --norm-vars false --do-concat false \
      --splice-opts "--left-context=5 --right-context=5" \
      $data/dt05_multi_$enhan ${gmmdir}_ali_dt05 $working_dir || exit 1
}

run

