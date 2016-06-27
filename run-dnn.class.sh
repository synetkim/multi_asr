#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

pdnndir=`pwd`/pdnn  
dataset=multi
working_dir=`pwd`/exp_pdnn.$1.$dataset
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
gpu=gpu`getUnusedGPU.sh`
num_pdfs=`cat $working_dir/num_pdfs.$dataset`
echo $gpu
echo "$infeat_dim x $num_pdfs => $outfeat_dim"

if [ ! -f $working_dir/dnn.ptr.done ]; then
  echo "SDA Pre-training"
  $cmd $working_dir/log/dnn.ptr.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_SdA.py --train-data "$working_dir/data.$dataset/train.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --nnet-spec "$infeat_dim:2048:2048:2048:2048:2048:2048:2048:$num_pdfs" \
                                    --1stlayer-reconstruct-activation "tanh" \
                                    --wdir $working_dir --param-output-file $working_dir/dnn.ptr \
                                    --ptr-layer-number 7 --epoch-number 2 || exit 1;
  touch $working_dir/dnn.ptr.done
fi

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  $cmd $working_dir/log/dnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_DNN.py --train-data "$working_dir/data.$dataset/train.*.scp,partition=3000m,random=false,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --valid-data "$working_dir/data.$dataset/valid.*.scp,partition=2000m,random=true,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --nnet-spec "$infeat_dim:2048:2048:2048:2048:2048:2048:2048:$num_pdfs" \
                                    --lrate "D:0.08:0.8:0.2,0.1:4" --momentum 0.9 \
                                    --ptr-file $working_dir/dnn.ptr --ptr-layer-number 7 \
                                    --param-output-file $working_dir/dnn.param --cfg-output-file $working_dir/dnn.cfg \
                                    --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
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
    python pdnn/cmds2/run_FeatExt_Kaldi.py --in_scp_file $tmpdir/input.$set.$n.scp \
                                    --out_ark_file $tmpdir/pred.$set.$n.ark \
                                    --nnet_param $working_dir/dnn.param \
                                    --nnet_cfg $working_dir/dnn.cfg \
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
    steps/compute_cmvn_stats.sh $outdir/ $outdir/log $outdir/ || exit 1;
    rm -rf $tmpdir

done

##################
# decoding 
##################
echo "Decoding ..."
for set in dt05_real et05_real; do
    gmmdir=exp.$dataset/tri3b_tr05_${dataset}_noisy
    graph_dir=$gmmdir/graph_tgpr_5k
    outdir=$working_dir/data-new/${set}_dnnenhan/
    steps_pdnn/decode_dnn_from_pdf.sh --nj 4 --cmd "$decode_cmd" \
    	$graph_dir $outdir/ ${gmmdir}_ali $working_dir/decode.dnn.$set &
    sleep 3;
done
wait;
echo "Done! "
rm -rf $working_dir/data-new/

