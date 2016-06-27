#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

function run() {

echo =====================================================================
enhan=$1
dataset=multi
out_data_dir=data-alignment.$enhan.$dataset
in_data_dir=data-fmllr  #doesn't important, only pick uttid
gmmdir=exp/tri3b_tr05_${dataset}_$enhan
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;
mkdir -p $out_data_dir/log

echo "dataset: $dataset"
echo "$num_pdfs" > $out_data_dir/num_pdfs.$dataset
echo "enhan: $enhan"
echo "in feat: $in_data_dir"
echo "in gmm: $gmmdir"
echo "output dir: $out_data_dir"
echo "Num of PDFs = $num_pdfs"
echo =====================================================================

nj=4

#steps/align_fmllr.sh --nj $nj \
#  data/tr05_${dataset}_$enhan data/lang exp/tri3b_tr05_${dataset}_$enhan exp/tri3b_tr05_${dataset}_${enhan}_ali || exit 1;
#steps/align_fmllr.sh --nj 4 \
#  data/dt05_${dataset}_$enhan data/lang exp/tri3b_tr05_${dataset}_$enhan exp/tri3b_tr05_${dataset}_${enhan}_ali_dt05 || exit 1;

#####################################################
#echo "Start new"
#dir=exp/tri4a_dnn_tr05_${dataset}_${enhan}
#alidir=exp/tri3b_tr05_${dataset}_${enhan}_ali
#alidir_cv=exp/tri3b_tr05_${dataset}_${enhan}_ali_dt05
#echo "	Using PDF targets from dirs '$alidir' '$alidir_cv'"
## define pdf-alignment rspecifiers
#labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
#labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
## 
#labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |" # for analyze-counts.
#labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
#lang=data/lang
#mkdir -p $dir/log
#analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
#echo "	copy the old transition model, will be needed by decoder"
#copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
#echo "	copy the tree"
#cp $alidir/tree $dir/tree || exit 1
#echo "	make phone counts for analysis"
#analyze-counts --verbose=1 --symbol-table=$lang/phones.txt "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log || exit 1
#echo "	output-dim"
#[ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')
#echo "	num_tgt: $num_tgt"
#utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
#echo "Finish new"
#####################################################

for set in dt05_${dataset} tr05_${dataset}; do
    cp -r $in_data_dir/${set}_${enhan} $out_data_dir/
	indata=$in_data_dir/${set}_${enhan}
	outdata=$out_data_dir/${set}_${enhan}
	aligndir=${gmmdir}_ali
    if [[ "$set" == "dt05_"* ]];then
		aligndir=${gmmdir}_ali_dt05
    fi
	soutdata=$outdata/split$nj
	sindata=$indata/split$nj
	[[ -d $soutdata && $outdata/feats.scp -ot $soutdata ]] || split_data.sh $outdata $nj || exit 1;
	[[ -d $sindata && $indata/feats.scp -ot $sindata ]] || split_data.sh $indata $nj || exit 1;
    rm $outdata/feats.scp
    rm $soutdata/*/feats.scp

	feats="ark,s,cs:copy-feats scp:$sindata/JOB/feats.scp ark:- |"

    $cmd JOB=1:$nj $out_data_dir/log/build_featwithalignment.$set.JOB.log \
            ./steps_featext/kaldi/build-alignment --every-nth-frame=1 $aligndir/final.mdl "ark:gunzip -c $aligndir/ali.JOB.gz|" \
            "$feats" ark,scp:$soutdata/JOB/feats.ark,$soutdata/JOB/feats.scp || exit 1;

    wait;
    cat $soutdata/*/feats.scp >>$outdata/feats.scp
    wc -l $outdata/feats.scp
done
}

#run close
#run beamform
run noisy
#run 6ch

