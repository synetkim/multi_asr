#!/bin/bash

# $1=data
# $2=feats

function run1(){
datadir=$1
list=""
list=$list" tr05_real_noisy1 dt05_real_noisy1"
list=$list" tr05_real_noisy3 dt05_real_noisy3"
list=$list" tr05_real_noisy4 dt05_real_noisy4"
list=$list" tr05_real_noisy5 dt05_real_noisy5"
list=$list" tr05_real_noisy6 dt05_real_noisy6"
list=$list" tr05_simu_noisy1 dt05_simu_noisy1"
list=$list" tr05_simu_noisy3 dt05_simu_noisy3"
list=$list" tr05_simu_noisy4 dt05_simu_noisy4"
list=$list" tr05_simu_noisy5 dt05_simu_noisy5"
list=$list" tr05_simu_noisy6 dt05_simu_noisy6"

list=$list" tr05_simu_wsj1_noisy1 "
list=$list" tr05_simu_wsj1_noisy3 "
list=$list" tr05_simu_wsj1_noisy4 "
list=$list" tr05_simu_wsj1_noisy5 "
list=$list" tr05_simu_wsj1_noisy6 "


for d in $list; do

echo "$datadir/$d "

sed -i 's/CH1//g' $datadir/$d/cmvn.scp
sed -i 's/CH1//g' $datadir/$d/feats.scp
sed -i 's/CH1//g' $datadir/$d/spk2utt
sed -i 's/CH1//g' $datadir/$d/text
sed -i 's/CH1//g' $datadir/$d/utt2spk
sed -i 's/CH1//g' $datadir/$d/wav.scp

sed -i 's/CH2//g' $datadir/$d/cmvn.scp
sed -i 's/CH2//g' $datadir/$d/feats.scp
sed -i 's/CH2//g' $datadir/$d/spk2utt
sed -i 's/CH2//g' $datadir/$d/text
sed -i 's/CH2//g' $datadir/$d/utt2spk
sed -i 's/CH2//g' $datadir/$d/wav.scp

sed -i 's/CH3//g' $datadir/$d/cmvn.scp
sed -i 's/CH3//g' $datadir/$d/feats.scp
sed -i 's/CH3//g' $datadir/$d/spk2utt
sed -i 's/CH3//g' $datadir/$d/text
sed -i 's/CH3//g' $datadir/$d/utt2spk
sed -i 's/CH3//g' $datadir/$d/wav.scp

sed -i 's/CH4//g' $datadir/$d/cmvn.scp
sed -i 's/CH4//g' $datadir/$d/feats.scp
sed -i 's/CH4//g' $datadir/$d/spk2utt
sed -i 's/CH4//g' $datadir/$d/text
sed -i 's/CH4//g' $datadir/$d/utt2spk
sed -i 's/CH4//g' $datadir/$d/wav.scp

sed -i 's/CH5//g' $datadir/$d/cmvn.scp
sed -i 's/CH5//g' $datadir/$d/feats.scp
sed -i 's/CH5//g' $datadir/$d/spk2utt
sed -i 's/CH5//g' $datadir/$d/text
sed -i 's/CH5//g' $datadir/$d/utt2spk
sed -i 's/CH5//g' $datadir/$d/wav.scp

sed -i 's/CH6//g' $datadir/$d/cmvn.scp
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

outdir=tr05_simu_wsj1_6ch
rm -rf $outdir
cp -rf $datadir/tr05_simu_wsj1_noisy1 $datadir/tr05_simu_wsj1_6ch
append-feats scp:`pwd`/$datadir/tr05_simu_wsj1_noisy1/feats.scp scp:$datadir/tr05_simu_wsj1_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13_wsj1.ark,`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp13_wsj1.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp13_wsj1.scp scp:$datadir/tr05_simu_wsj1_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134_wsj1.ark,`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp134_wsj1.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp134_wsj1.scp scp:$datadir/tr05_simu_wsj1_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345_wsj1.ark,`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp1345_wsj1.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_wsj1_6ch/tmp1345_wsj1.scp scp:$datadir/tr05_simu_wsj1_noisy6/feats.scp ark,scp:`pwd`/$feats/tr05_simu_wsj1_6ch.13456.ark,`pwd`/$datadir/tr05_simu_wsj1_6ch/feats.scp ||exit 1
rm `pwd`/$datadir/tr05_simu_wsj1_6ch/tmp*
rm `pwd`/$feats/tmp*

outdir=tr05_simu_6ch
rm -rf $outdir
cp -rf $datadir/tr05_simu_noisy1 $datadir/tr05_simu_6ch
append-feats scp:`pwd`/$datadir/tr05_simu_noisy1/feats.scp scp:$datadir/tr05_simu_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/tr05_simu_6ch/tmp13.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_6ch/tmp13.scp scp:$datadir/tr05_simu_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/tr05_simu_6ch/tmp134.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_6ch/tmp134.scp scp:$datadir/tr05_simu_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/tr05_simu_6ch/tmp1345.scp ||exit 1
append-feats scp:`pwd`/$datadir/tr05_simu_6ch/tmp1345.scp scp:$datadir/tr05_simu_noisy6/feats.scp ark,scp:`pwd`/$feats/tr05_simu_6ch.13456.ark,`pwd`/$datadir/tr05_simu_6ch/feats.scp ||exit 1
rm `pwd`/$datadir/tr05_simu_6ch/tmp*
rm `pwd`/$feats/tmp*

outdir=dt05_simu_6ch
rm -rf $outdir
cp -rf $datadir/dt05_simu_noisy1 $datadir/dt05_simu_6ch
append-feats scp:`pwd`/$datadir/dt05_simu_noisy1/feats.scp scp:$datadir/dt05_simu_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/dt05_simu_6ch/tmp13.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_simu_6ch/tmp13.scp scp:$datadir/dt05_simu_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/dt05_simu_6ch/tmp134.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_simu_6ch/tmp134.scp scp:$datadir/dt05_simu_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/dt05_simu_6ch/tmp1345.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_simu_6ch/tmp1345.scp scp:$datadir/dt05_simu_noisy6/feats.scp ark,scp:`pwd`/$feats/dt05_simu_6ch.13456.ark,`pwd`/$datadir/dt05_simu_6ch/feats.scp || exit 1
rm `pwd`/$datadir/dt05_simu_6ch/tmp*
rm `pwd`/$feats/tmp*


outdir=tr05_real_6ch
rm -rf $outdir
cp -rf $datadir/tr05_real_noisy1 $datadir/tr05_real_6ch
append-feats scp:`pwd`/$datadir/tr05_real_noisy1/feats.scp scp:$datadir/tr05_real_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/tr05_real_6ch/tmp13.scp || exit 1
append-feats scp:`pwd`/$datadir/tr05_real_6ch/tmp13.scp scp:$datadir/tr05_real_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/tr05_real_6ch/tmp134.scp || exit 1
append-feats scp:`pwd`/$datadir/tr05_real_6ch/tmp134.scp scp:$datadir/tr05_real_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/tr05_real_6ch/tmp1345.scp || exit 1
append-feats scp:`pwd`/$datadir/tr05_real_6ch/tmp1345.scp scp:$datadir/tr05_real_noisy6/feats.scp ark,scp:`pwd`/$feats/tr05_real_6ch.13456.ark,`pwd`/$datadir/tr05_real_6ch/feats.scp || exit 1
rm `pwd`/$datadir/tr05_real_6ch/tmp*
rm `pwd`/$feats/tmp*

outdir=dt05_real_6ch
rm -rf $outdir
cp -rf $datadir/dt05_real_noisy1 $datadir/dt05_real_6ch
append-feats scp:`pwd`/$datadir/dt05_real_noisy1/feats.scp scp:$datadir/dt05_real_noisy3/feats.scp ark,scp:`pwd`/$feats/tmp13.ark,`pwd`/$datadir/dt05_real_6ch/tmp13.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_real_6ch/tmp13.scp scp:$datadir/dt05_real_noisy4/feats.scp ark,scp:`pwd`/$feats/tmp134.ark,`pwd`/$datadir/dt05_real_6ch/tmp134.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_real_6ch/tmp134.scp scp:$datadir/dt05_real_noisy5/feats.scp ark,scp:`pwd`/$feats/tmp1345.ark,`pwd`/$datadir/dt05_real_6ch/tmp1345.scp || exit 1
append-feats scp:`pwd`/$datadir/dt05_real_6ch/tmp1345.scp scp:$datadir/dt05_real_noisy6/feats.scp ark,scp:`pwd`/$feats/dt05_real_6ch.13456.ark,`pwd`/$datadir/dt05_real_6ch/feats.scp || exit 1
rm `pwd`/$datadir/dt05_real_6ch/tmp*
rm `pwd`/$feats/tmp*
}

run1 $1
run2 $1 $2

