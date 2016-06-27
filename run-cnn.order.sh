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

#fbank 1000 + 13
# ch1ch3ch4ch5ch6, ..., ch1ch3ch4ch5ch6
# ch1x5frame, ch2x5frame, ..., ch5x5frame
nlabel=39
nlabel=12

if [ ! -f $working_dir/changeorder.fine.done ]; then
  echo "Start. change order"
  $cmd JOB=1:$nj $working_dir/log/order_train.JOB.log \
   select-feats 0-39,200-239,400-439,600-639,800-839,40-79,240-279,440-479,640-679,840-879,80-119,280-319,480-519,680-719,880-919,120-159,320-359,520-559,720-759,920-959,160-199,360-399,560-599,760-799,960-999,1000-10$nlabel \
   scp:exp_pdnn/data.regressionfeat.closemfcc/train.JOB.scp ark,scp:exp_pdnn/data.regressionfeat.closemfcc/order.train.JOB.ark,exp_pdnn/data.regressionfeat.closemfcc/order.train.JOB.scp || exit 1;

  $cmd JOB=1:$testnj $working_dir/log/order_valid.JOB.log \
   select-feats 0-39,200-239,400-439,600-639,800-839,40-79,240-279,440-479,640-679,840-879,80-119,280-319,480-519,680-719,880-919,120-159,320-359,520-559,720-759,920-959,160-199,360-399,560-599,760-799,960-999,1000-10$nlabel \
   scp:exp_pdnn/data.regressionfeat.closemfcc/valid.JOB.scp ark,scp:exp_pdnn/data.regressionfeat.closemfcc/order.valid.JOB.ark,exp_pdnn/data.regressionfeat.closemfcc/order.valid.JOB.scp || exit 1;
  wait;
  echo "Done. change order"
  touch $working_dir/changeorder.fine.done
fi

if [ ! -f $working_dir/cnn.fine.done ]; then
  echo "Fine-tuning CNN"
  $cmd $working_dir/log/cnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_CNNV.py --train-data "$working_dir/data.regressionfeat.closemfcc/order.train.*.scp,partition=3000m,random=true,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --valid-data "$working_dir/data.regressionfeat.closemfcc/order.valid.*.scp,partition=3000m,random=true,stream=true,target_dim=$outfeat_dim,extra_dim=0" \
                                    --conv-nnet-spec "5x5x40:256,2x10,p2x3:256,1x3,p1x1,f"  \
                                    --nnet-spec "1024:1024:$outfeat_dim" \
                                    --lrate "MD:0.08:0.8:0.0004,0.00001:6" --momentum 0.9 \
                                    --batch-size 32 \
                                    --l2-reg 0.2 \
                                    --param-output-file $working_dir/cnn.param --cfg-output-file $working_dir/cnn.cfg \
                                    --wdir $working_dir --kaldi-output-file $working_dir/cnn.nnet || exit 1;
  touch $working_dir/cnn.fine.done
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
	select-feats 0-39,200-239,400-439,600-639,800-839,40-79,240-279,440-479,640-679,840-879,80-119,280-319,480-519,680-719,880-919,120-159,320-359,520-559,720-759,920-959,160-199,360-399,560-599,760-799,960-999 \
			scp:$tmpdir/input.$set.$n.scp ark,scp:$tmpdir/order.input.$set.$n.ark,$tmpdir/order.input.$set.$n.scp
    gpu=gpu`getUnusedGPU.sh`

    $cmd $working_dir/log/feat.fine.$set.$n.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds2/run_CnnFeat_sy.py --in_scp_file $tmpdir/order.input.$set.$n.scp \
            --out_ark_file $tmpdir/pred.$set.$n.ark \
            --cnn_param_file $working_dir/cnn.param \
            --activation "sigmoid" \
            --cnn_cfg_file $working_dir/cnn.cfg \
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

