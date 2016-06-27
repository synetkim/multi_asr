#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

pdnndir=`pwd`/pdnn  
gpu=gpu`getUnusedGPU.sh`
working_dir=`pwd`/exp_pdnn
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
infeat_dim=$(cat $working_dir/infeat_dim| awk '{print $2}')
outfeat_dim=$(cat $working_dir/outfeat_dim| awk '{print $2}')

echo $gpu
echo "$infeat_dim x $outfeat_dim"

if [ ! -f $working_dir/extra.fine.done ]; then
  rm $working_dir/data.regressionfeat.closemfcc/extra_train.*.ark
  rm $working_dir/data.regressionfeat.closemfcc/extra_valid.*.ark

  model_feat="subset-feats --include=$working_dir/data.regressionfeat.closemfcc/train.JOB.scp scp:data-phasediff/tr05_real_6ch/feats.scp ark:- | paste-feats scp:$working_dir/data.regressionfeat.closemfcc/train.JOB.scp ark:- ark,scp:$working_dir/data.regressionfeat.closemfcc/extra_with_train.JOB.ark,$working_dir/data.regressionfeat.closemfcc/extra_with_train.JOB.scp "
  $cmd JOB=1:$nj $working_dir/log/build_extra_train.JOB.log $model_feat || exit 1;

  model_feat="subset-feats --include=$working_dir/data.regressionfeat.closemfcc/valid.JOB.scp scp:data-phasediff/dt05_real_6ch/feats.scp ark:- | paste-feats scp:$working_dir/data.regressionfeat.closemfcc/valid.JOB.scp ark:- ark,scp:$working_dir/data.regressionfeat.closemfcc/extra_with_valid.JOB.ark,$working_dir/data.regressionfeat.closemfcc/extra_with_valid.JOB.scp "
  $cmd JOB=1:$nj $working_dir/log/build_extra_valid.JOB.log $model_feat || exit 1;

  touch $working_dir/extra.fine.done
fi

if [ ! -f $working_dir/phaseattendlstm.fine.done ]; then
  echo "Fine-tuning Phase-based attention LSTMV"
  $cmd $working_dir/log/phaseattendlstm.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_Phase_ATTEND_LSTM.py --train-data "$working_dir/data.regressionfeat.closemfcc/t.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --valid-data "$working_dir/data.regressionfeat.closemfcc/v.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --lstm-nnet-spec "$infeat_dim:256" \
                                    --nnet-spec "256:$outfeat_dim" \
                                    --extra-nnet-spec "256:256:256:20"  \
                                    --activation "rectifier" \
                                    --max-col-norm 1 \
                                    --lrate "MD:0.8:0.8:0.0004,0.00001:4" --momentum 0.9 \
                                    --l2-reg 0.2 \
                                    --batch-size 8 \
                                    --param-output-file $working_dir/phaseattendlstm.param --cfg-output-file $working_dir/phaseattendlstm.cfg \
                                    --wdir $working_dir || exit 1;
  touch $working_dir/phaseattendlstm.fine.done
                                    #--dropout-factor "0.2,0.2" \
fi

echo "Generate Features"
function forward()
{
    set=$1
    tmpdir=$2
    n=$3
    sdata_in=$4

    infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_in/$n/utt2spk scp:$sdata_in/$n/cmvn.scp scp:$sdata_in/$n/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
    copy-feats "$infeats" ark,scp:$tmpdir/input.$set.$n.ark,$tmpdir/input.$set.$n.scp
    subset-feats --include=$tmpdir/input.$set.$n.scp scp:$extra_data_in/feats.scp \
                ark,scp:$tmpdir/extra_input.$set.$n.ark,$tmpdir/extra_input.$set.$n.scp
    gpu=gpu`getUnusedGPU.sh`

    $cmd $working_dir/log/feat.fine.$set.$n.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
	python pdnn/cmds2/run_PhaseATTENDLSTMFeat_sy.py --in_scp_file $tmpdir/input.$set.$n.scp \
                                    --out_ark_file $tmpdir/pred.$set.$n.ark \
            						--extra_in_scp_file $tmpdir/extra_input.$set.$n.scp \
                                    --lstm_param_file $working_dir/phaseattendlstm.param \
                                    --activation "rectifier" \
                                    --lstm_cfg_file $working_dir/phaseattendlstm.cfg \
                                    --layer_index -1 || exit 1;

    copy-feats ark:$tmpdir/pred.$set.$n.ark ark,scp:$outdir/feats_predict.${set}_${input_data_type}.$n.ark,$outdir/feats_predict.${set}_${input_data_type}.$n.scp || exit 1;
}

for set in et05_real tr05_real dt05_real; do
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
        forward $set $tmpdir $n "$sdata_in" || exit 1;
    done
    wait;

    for n in $(seq $nj); do
        cat $outdir/feats_predict.${set}_${input_data_type}.$n.scp >> $outdir/feats.scp
    done
    cp $input_dir/${set}_${input_data_type}/spk2utt $outdir/
    cp $input_dir/${set}_${input_data_type}/utt2spk $outdir/
    cp $input_dir/${set}_${input_data_type}/text $outdir/
    steps/compute_cmvn_stats.sh $outdir/ $outdir/log $outdir/ || exit 1;
    rm -rf $tmpdir
done

echo "Done! Generate Features"
  

