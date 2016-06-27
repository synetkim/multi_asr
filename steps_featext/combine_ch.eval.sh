#!/bin/bash

# $1=data
# $2=feats

function run1(){
datadir=$1

list=""
list=$list" et05_real_noisy1"
list=$list" et05_real_noisy2"
list=$list" et05_real_noisy3"
list=$list" et05_real_noisy4"
list=$list" et05_real_noisy5"
list=$list" et05_real_noisy6"

list=$list" et05_simu_noisy1"
list=$list" et05_simu_noisy2"
list=$list" et05_simu_noisy3"
list=$list" et05_simu_noisy4"
list=$list" et05_simu_noisy5"
list=$list" et05_simu_noisy6"

for d in $list; do

echo "$datadir/$d "

sed -i 's/CH1//g' $datadir/$d/feats.scp
sed -i 's/CH1//g' $datadir/$d/spk2utt
sed -i 's/CH1//g' $datadir/$d/text
sed -i 's/CH1//g' $datadir/$d/utt2spk
sed -i 's/CH1//g' $datadir/$d/wav.scp

sed -i 's/CH2//g' $datadir/$d/feats.scp
sed -i 's/CH2//g' $datadir/$d/spk2utt
sed -i 's/CH2//g' $datadir/$d/text
sed -i 's/CH2//g' $datadir/$d/utt2spk
sed -i 's/CH2//g' $datadir/$d/wav.scp

sed -i 's/CH3//g' $datadir/$d/feats.scp
sed -i 's/CH3//g' $datadir/$d/spk2utt
sed -i 's/CH3//g' $datadir/$d/text
sed -i 's/CH3//g' $datadir/$d/utt2spk
sed -i 's/CH3//g' $datadir/$d/wav.scp

sed -i 's/CH4//g' $datadir/$d/feats.scp
sed -i 's/CH4//g' $datadir/$d/spk2utt
sed -i 's/CH4//g' $datadir/$d/text
sed -i 's/CH4//g' $datadir/$d/utt2spk
sed -i 's/CH4//g' $datadir/$d/wav.scp

sed -i 's/CH5//g' $datadir/$d/feats.scp
sed -i 's/CH5//g' $datadir/$d/spk2utt
sed -i 's/CH5//g' $datadir/$d/text
sed -i 's/CH5//g' $datadir/$d/utt2spk
sed -i 's/CH5//g' $datadir/$d/wav.scp

sed -i 's/CH6//g' $datadir/$d/feats.scp
sed -i 's/CH6//g' $datadir/$d/spk2utt
sed -i 's/CH6//g' $datadir/$d/text
sed -i 's/CH6//g' $datadir/$d/utt2spk
sed -i 's/CH6//g' $datadir/$d/wav.scp
done
}

function run2(){
datadir=$1
feats=$2

for x in et05_real et05_simu; do
  outdir=$datadir/${x}_6ch
  rm -rf $outdir
  mkdir -p $outdir
  cp $datadir/${x}_noisy1/spk2utt $outdir/
  cp $datadir/${x}_noisy1/text $outdir/
  cp $datadir/${x}_noisy1/utt2spk $outdir/
  append-feats scp:`pwd`/$datadir/${x}_noisy1/feats.scp scp:$datadir/${x}_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/${x}_6ch/tmp13.scp ||exit 1
  append-feats scp:`pwd`/$datadir/${x}_6ch/tmp13.scp scp:$datadir/${x}_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/${x}_6ch/tmp134.scp ||exit 1
  append-feats scp:`pwd`/$datadir/${x}_6ch/tmp134.scp scp:$datadir/${x}_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/${x}_6ch/tmp1345.scp ||exit 1
  append-feats scp:`pwd`/$datadir/${x}_6ch/tmp1345.scp scp:$datadir/${x}_noisy6/feats.scp ark,scp:`pwd`/$feats/${x}_6ch.13456.ark,`pwd`/$datadir/${x}_6ch/feats.scp ||exit 1
  rm `pwd`/$datadir/${x}_6ch/tmp*
  rm `pwd`/$feats/tmp*
done

}

run1 $1
run2 $1 $2

