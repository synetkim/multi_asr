#!/bin/bash

. ./path.sh
. ./cmd.sh 

chime3_data=/data-local/corpora/CHiME3/
featdir=data-fft

function first()
{
enhancement_data=xx
enhancement_method=noisy5

utils/combine_data.sh $featdir/tr05_multi_noisy5 $featdir/tr05_simu_noisy5 $featdir/tr05_real_noisy5
utils/combine_data.sh $featdir/dt05_multi_noisy5 $featdir/dt05_simu_noisy5 $featdir/dt05_real_noisy5
local/run_gmm.newdata.sh $enhancement_method $enhancement_data $featdir

mv exp/tri3b_tr05_multi_noisy5 exp/tri3b_tr05_multi_dnnenhan
mv exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_dt05_real_noisy5 exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_dt05_real_dnnenhan
mv exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_dt05_simu_noisy5 exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_dt05_simu_dnnenhan
mv exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_et05_real_noisy5 exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_et05_real_dnnenhan
mv exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_et05_simu_noisy5 exp/tri3b_tr05_multi_dnnenhan/decode_tgpr_5k_et05_simu_dnnenhan

rm exp/tri3*ali -rf
mv exp exp.$featdir
}
function second()
{
rm -rf exp/tri3b*
mkdir -p exp
cp -rf /data-local3/sykim/mend/exp.$featdir/tri3b_tr05_multi_dnnenhan ./exp/
enhancement_data=xx
enhancement_method=dnnenhan
local/run_dnn_fmllr.newdata.sh $enhancement_method $enhancement_data exp_pdnn/data-new 
}

second
