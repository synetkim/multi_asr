#!/bin/bash
#
# $1=data
# $2=feats
# $3=list

echo "$3"

function run2(){
datadir=$1
feats=$2

for x in tr05_simu dt05_simu tr05_real dt05_real et05_real; do
  outdir=$datadir/${x}_6ch
  rm -rf $outdir
  mkdir -p $outdir
  cp $datadir/${x}_noisy1/spk2utt $outdir/
  cp $datadir/${x}_noisy1/text $outdir/
  cp $datadir/${x}_noisy1/utt2spk $outdir/
  paste-feats scp:`pwd`/$datadir/${x}_noisy1/feats.scp scp:$datadir/${x}_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/${x}_6ch/tmp13.scp ||exit 1
  paste-feats scp:`pwd`/$datadir/${x}_6ch/tmp13.scp scp:$datadir/${x}_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/${x}_6ch/tmp134.scp ||exit 1
  paste-feats scp:`pwd`/$datadir/${x}_6ch/tmp134.scp scp:$datadir/${x}_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/${x}_6ch/tmp1345.scp ||exit 1
  paste-feats scp:`pwd`/$datadir/${x}_6ch/tmp1345.scp scp:$datadir/${x}_noisy6/feats.scp ark,scp:`pwd`/$feats/${x}_6ch.13456.ark,`pwd`/$datadir/${x}_6ch/feats.scp ||exit 1
  rm `pwd`/$datadir/${x}_6ch/tmp*
  rm `pwd`/$feats/tmp*
done

}

run2 $1 $2

