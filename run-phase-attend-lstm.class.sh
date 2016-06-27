#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

pdnndir=`pwd`/pdnn  
dataset=multi
working_dir=`pwd`/exp_pdnn.6ch.$dataset
input_dir=`cat $working_dir/input_dir`
input_data_dir=`cat $working_dir/input_data_dir`
input_data_type=`cat $working_dir/input_data_type`
target_data_dir=`cat $working_dir/target_data_dir`
target_data_type=`cat $working_dir/target_data_type`
nj=`cat $working_dir/nj`
testnj=`cat $working_dir/testnj`
splice_opts=`cat $working_dir/splice_opts`
split_opts=`cat $working_dir/split_opts`
norm_vars=`cat $working_dir/norm_vars`
cmvn=`cat $working_dir/cmvn`
infeat_dim=$(cat $working_dir/infeat_dim| awk '{print $2}')
outfeat_dim=$(cat $working_dir/outfeat_dim| awk '{print $2}')
gpu=gpu`getUnusedGPU.sh`
num_pdfs=`cat $working_dir/num_pdfs.$dataset`
phase_dim=2560
phase_feat_select="256-1279,1792-2559,3328-3839,4864-5119"
phase_data_id=10
#phase_dim=1024
#phase_feat_select="3840-4607,4864-5119"
#phase_data_id=5
echo $gpu
echo "$infeat_dim x $num_pdfs + $phase_dim => $outfeat_dim"

if [ ! -f $working_dir/$phase_data_id.extra.fine.done ]; then
  rm $working_dir/data.$dataset/extra_train.*.ark
  rm $working_dir/data.$dataset/extra_valid.*.ark

  model_feat="select-feats $phase_feat_select scp:data-phasediff/tr05_${dataset}_6ch/feats.scp ark:- | subset-feats --include=$working_dir/data.$dataset/train.JOB.scp ark:- ark:- | paste-feats scp:$working_dir/data.$dataset/train.JOB.scp ark:- ark,scp:$working_dir/data.$dataset/$phase_data_id.extra_with_train.JOB.ark,$working_dir/data.$dataset/$phase_data_id.extra_with_train.JOB.scp "
  $cmd JOB=1:$nj $working_dir/log/build_extra_train.JOB.log $model_feat || exit 1;

  model_feat="select-feats $phase_feat_select scp:data-phasediff/dt05_${dataset}_6ch/feats.scp ark:- | subset-feats --include=$working_dir/data.$dataset/valid.JOB.scp ark:- ark:- | paste-feats scp:$working_dir/data.$dataset/valid.JOB.scp ark:- ark,scp:$working_dir/data.$dataset/$phase_data_id.extra_with_valid.JOB.ark,$working_dir/data.$dataset/$phase_data_id.extra_with_valid.JOB.scp "
  $cmd JOB=1:$nj $working_dir/log/build_extra_valid.JOB.log $model_feat || exit 1;

  touch $working_dir/$phase_data_id.extra.fine.done
fi

if [ ! -f $working_dir/phaseattendlstm.fine.done ]; then
  echo "Fine-tuning PhaseAttendLSTMV"
  $cmd $working_dir/log/phaseattendlstm.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_Phase_ATTEND_LSTM.py --train-data "$working_dir/data.$dataset/$phase_data_id.extra_with_train.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=$phase_dim" \
                                    --valid-data "$working_dir/data.$dataset/$phase_data_id.extra_with_valid.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=$phase_dim" \
                                    --lstm-nnet-spec "$infeat_dim:512" \
                                    --nnet-spec "$num_pdfs" \
                                    --lrate "MD:0.4:0.4:0.2,0.0001:2" \
                                    --batch-size 32 \
                                    --param-output-file $working_dir/phaseattendlstm.param --cfg-output-file $working_dir/phaseattendlstm.cfg \
                                    --wdir $working_dir || exit 1;
  touch $working_dir/phaseattendlstm.fine.done
fi

echo "Generate Features"
function forward()
{
    set=$1
    tmpdir=$2
    n=$3
    sdata_in=$4
    extra_data_in=data-phasediff/${set}_6ch

    if [ "$cmvn" == "true" ];then
        echo "CMVN"
        infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_in/$n/utt2spk scp:$sdata_in/$n/cmvn.scp scp:$sdata_in/$n/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
    else
        echo "no CMVN"
        infeats="ark,s,cs:splice-feats $splice_opts scp:$sdata_in/$n/feats.scp ark:- |"
    fi
    copy-feats "$infeats" ark,scp:$tmpdir/input.$set.$n.ark,$tmpdir/input.$set.$n.scp
    select-feats "$phase_feat_select" scp:$extra_data_in/feats.scp ark:- | \
        subset-feats --include=$tmpdir/input.$set.$n.scp ark:-  \
        ark,scp:$tmpdir/$phase_data_id.extra_input.$set.$n.ark,$tmpdir/$phase_data_id.extra_input.$set.$n.scp

    gpu=gpu`getUnusedGPU.sh`

    $cmd $working_dir/log/feat.fine.$set.$n.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds2/run_PhaseATTENDLSTMFeat.py --in_scp_file $tmpdir/input.$set.$n.scp \
                                    --extra_in_scp_file $tmpdir/$phase_data_id.extra_input.$set.$n.scp \
                                    --out_ark_file $tmpdir/pred.$set.$n.ark \
                                    --activation "sigmoid" \
                                    --lstm_param_file $working_dir/phaseattendlstm.param \
                                    --lstm_cfg_file $working_dir/phaseattendlstm.cfg \
                                    --layer_index -1 || exit 1;

    copy-feats ark:$tmpdir/pred.$set.$n.ark ark,scp:$outdir/feats_predict.${set}_${input_data_type}.$n.ark,$outdir/feats_predict.${set}_${input_data_type}.$n.scp || exit 1;
}

for set in et05_real dt05_real; do
	nj=4
    outdir=$working_dir/data-new/${set}_dnnenhan/
    tmpdir=$working_dir/data-new/${set}_tmp/
    echo "inputdir = $input_dir"
    echo "outdir = $outdir"
    mkdir -p $outdir
    mkdir -p $tmpdir

    sdata_in=$input_dir/${set}_${input_data_type}/split$nj
    [[ -d $sdata_in && $input_dir/${set}_${input_data_type}/feats.scp -ot $sdata_in ]] || split_data.sh $split_opts $input_dir/${set}_${input_data_type} $nj || exit 1;

    echo "Extract Feature Kaldi format"
    rm $outdir/feats.scp
    for n in $(seq $nj); do
        forward $set $tmpdir $n "$sdata_in" &
		sleep 5;
    done
    wait;

    for n in $(seq $nj); do
        cat $outdir/feats_predict.${set}_${input_data_type}.$n.scp >> $outdir/feats.scp
    done
    cp $input_dir/${set}_${input_data_type}/spk2utt $outdir/
    cp $input_dir/${set}_${input_data_type}/utt2spk $outdir/
    cp $input_dir/${set}_${input_data_type}/text $outdir/
    if [ "$cmvn" == "true" ];then
        echo "CMVN"
        steps/compute_cmvn_stats.sh $outdir/ $outdir/log $outdir/ || exit 1;
    fi
    rm -rf $tmpdir
done

echo "Done! Generate Features"

##################
# decoding 
##################
echo "Decoding ..."
for set in dt05_real et05_real; do
    gmmdir=exp/tri3b_tr05_${dataset}_noisy
    graph_dir=$gmmdir/graph_tgpr_5k
    outdir=$working_dir/data-new/${set}_dnnenhan/
    steps_pdnn/decode_dnn_from_pdf.sh --nj 4 --cmd "$decode_cmd" \
    	$graph_dir $outdir/ ${gmmdir}_ali $working_dir/decode.phaseattlstm.$dataset.$set &
    sleep 3;
done
wait;
rm $working_dir/data-new -rf
echo "Done! "
 

