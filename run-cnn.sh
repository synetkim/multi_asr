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
splice_opts=`cat $working_dir/splice_opts`
split_opts=`cat $working_dir/split_opts`
norm_vars=`cat $working_dir/norm_vars`
infeat_dim=$(cat $working_dir/infeat_dim| awk '{print $2}')
outfeat_dim=$(cat $working_dir/outfeat_dim| awk '{print $2}')

echo $gpu
echo "$infeat_dim x $outfeat_dim"

if [ ! -f $working_dir/cnn.fine.done ]; then
  echo "Fine-tuning CNN"
  $cmd $working_dir/log/cnn.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds/run_CNNV.py --train-data "train.1.scp,partition=3000m,random=true,stream=true,target_dim=13,extra_dim=0" \
                                    --valid-data "valid.1.scp,partition=3000m,random=true,stream=true,target_dim=13,extra_dim=0" \
                                    --conv-nnet-spec "5x5x40:256,2x10,p2x3:256,1x3,p1x1,f"  \
                                    --nnet-spec "1024:1024:$outfeat_dim" \
                                    --lrate "MD:0.08:0.6:0.04,0.001:4" --momentum 0.9 \
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
	gpu=gpu`getUnusedGPU.sh`

	$cmd $working_dir/log/feat.fine.$set.$n.log \
	export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
	export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
	python pdnn/cmds2/run_CnnFeat_sy.py --in_scp_file $tmpdir/input.$set.$n.scp \
			--out_ark_file $tmpdir/pred.$set.$n.ark \
			--cnn_param_file $working_dir/cnn.param \
			--activation "sigmoid" \
			--cnn_cfg_file $working_dir/cnn.cfg \
			--layer_index -1 || exit 1;

	copy-feats ark:$tmpdir/pred.$set.$n.ark ark,scp:$outdir/feats_predict.${set}_${input_data_type}.$n.ark,$outdir/feats_predict.${set}_${input_data_type}.$n.scp || exit 1;
}

for set in et05_real dt05_real tr05_real ; do
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
#utils/combine_data.sh $working_dir/data-new/tr05_multi_dnnenhan $working_dir/data-new/tr05_simu_dnnenhan $working_dir/data-new/tr05_real_dnnenhan
#utils/combine_data.sh $working_dir/data-new/dt05_multi_dnnenhan $working_dir/data-new/dt05_simu_dnnenhan $working_dir/data-new/dt05_real_dnnenhan

echo "Done! Generate Features"



