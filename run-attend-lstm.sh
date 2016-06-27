#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

pdnndir=`pwd`/pdnn  
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

echo "$infeat_dim x $outfeat_dim"
gpu=gpu`getUnusedGPU.sh`
if [ ! -f $working_dir/attendlstm.fine.done ]; then
  echo "Fine-tuning AttendLSTMV"
  $cmd $working_dir/log/attendlstm.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_ATTEND_LSTMV.py --train-data "exp_pdnn/data/train.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --valid-data "exp_pdnn/data/valid.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --lstm-nnet-spec "$infeat_dim:256:13" \
                                    --nnet-spec "$outfeat_dim" \
                                    --activation "rectifier" \
                                    --lrate "MD:1.2:0.5:0.004,0.004:3" \
                                    --batch-size 32 \
                                    --param-output-file $working_dir/attendlstm.param --cfg-output-file $working_dir/attendlstm.cfg \
                                    --wdir $working_dir || exit 1;
  touch $working_dir/attendlstm.fine.done
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
    gpu=gpu`getUnusedGPU.sh`

    $cmd $working_dir/log/feat.fine.$set.$n.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
	python pdnn/cmds2/run_ATTENDLSTMFeat_sy.py --in_scp_file $tmpdir/input.$set.$n.scp \
                                    --out_ark_file $tmpdir/pred.$set.$n.ark \
                                    --lstm_param_file $working_dir/attendlstm.param \
                                    --activation "rectifier" \
                                    --lstm_cfg_file $working_dir/attendlstm.cfg \
                                    --layer_index -1 || exit 1;

    copy-feats ark:$tmpdir/pred.$set.$n.ark ark,scp:$outdir/feats_predict.${set}_${input_data_type}.$n.ark,$outdir/feats_predict.${set}_${input_data_type}.$n.scp || exit 1;
}

for set in et05_real tr05_real dt05_real; do
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
		sleep 3;
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
  

