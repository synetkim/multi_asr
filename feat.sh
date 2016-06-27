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

echo "Generate Features"

# generate feature representations with the CNN nosoft model
for set in tr05_simu tr05_real dt05_simu dt05_real; do
    outdir=$working_dir/data-new/${set}_dnnenhan/
    tmpdir=$working_dir/data-new/${set}_tmp/
    echo "inputdir = $input_dir"
    echo "outdir = $outdir"
    mkdir -p $outdir
    mkdir -p $tmpdir

    sdata_in=$input_dir/${set}_${input_data_type}/split$nj
    [[ -d $sdata_in && $input_dir/${set}_${input_data_type}/feats.scp -ot $sdata_in ]] || split_data.sh $split_opts $input_dir/${set}_${input_data_type} $nj || exit 1;
    infeats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata_in/JOB/utt2spk scp:$sdata_in/JOB/cmvn.scp scp:$sdata_in/JOB/feats.scp ark:- |splice-feats $splice_opts ark:- ark:- |"
    $cmd JOB=1:1 $working_dir/log/copyfeat.JOB.log \
		   copy-feats "$infeats" ark,scp:$tmpdir/$set.JOB.ark,$tmpdir/$set.JOB.scp

    echo "Extract Feature Kaldi format"
    $cmd $working_dir/log/feat.fine.log \
    export PYTHONPATH=$PYTHONPATH:`pwd`/pdnn/ \; \
    export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
    python pdnn/cmds2/run_FeatExt_Kaldi_sy.py --in_scp_file $tmpdir/$set.1.scp \
                                    --out_ark_file $outdir/test.ark \
                                    --nnet_param $working_dir/dnn.param \
                                    --nnet_cfg $working_dir/dnn.cfg \
                                    --layer_index -1 || exit 1;
done

echo "Done! Generate Features"



