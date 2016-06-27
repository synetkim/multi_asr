#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

path_result=../mend.mon.1/exp_pdnn.6ch.real/phaseattendlstm
path_param=$path_result.param
path_cfg=$path_result.cfg

head -1 analyze/input.et05_real.1.scp >analyze/et5.i.scp
tail -1 analyze/input.et05_real.1.scp >>analyze/et5.i.scp
copy-feats scp:analyze/et5.i.scp ark,scp:analyze/i.ark,analyze/i.scp
head -1 analyze/5.extra_input.et05_real.1.scp >analyze/et5.e.scp
tail -1 analyze/5.extra_input.et05_real.1.scp >>analyze/et5.e.scp
copy-feats scp:analyze/et5.e.scp ark,scp:analyze/e.ark,analyze/e.scp

echo "Draw Features"
function forward()
{
    gpu=gpu`getUnusedGPU.sh`
    echo $gpu

    $cmd analyze/log/feat.draw.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds2/run_draw_attend.py --in_scp_file analyze/i.scp \
                                    --extra_in_scp_file analyze/e.scp \
                                    --out_ark_file analyze/pred.ark \
                                    --lstm_param_file $path_param \
                                    --lstm_cfg_file $path_cfg \
                                    --layer_index -1 || exit 1;

    copy-feats ark:analyze/pred.ark ark,scp:analyze/p.ark,analyze/p.scp || exit 1;
}

for set in et05_real; do
    echo "Extract Feature Kaldi format"
    forward $set 0 0 0 || exit 1;
done
echo "Done! Draw Features"
 

