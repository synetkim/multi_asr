


function run() {
datadir=$1
echo "$datadir"
sed -i 's/CH0//g' $datadir/*.scp
sed -i 's/CH0//g' $datadir/*utt*
sed -i 's/CH0//g' $datadir/text
sed -i 's/CH0//g' $datadir/*dir*

sed -i 's/data-local1\/sykim\/mend.fordata/data-local\/suyoung1\/data.xcorr/g' $datadir/*.scp
sed -i 's/data-local1\/sykim\/mend.fordata/data-local\/suyoung1\/data.xcorr/g' $datadir/*utt*
sed -i 's/data-local1\/sykim\/mend.fordata/data-local\/suyoung1\/data.xcorr/g' $datadir/text
sed -i 's/data-local1\/sykim\/mend.fordata/data-local\/suyoung1\/data.xcorr/g' $datadir/*dir*
sed -i 's/data-local1\/sykim\/mend.fordata\/exp_pdnn/\data.fbankmfcc.real/data-local\/suyoung1\/data.xcorr\/exp_pdnn/\data.mfcc/g' $datadir/*scp
[ -f $datadir/feats.scp ] && feat-to-dim scp:$datadir/feats.scp -
}

run "data-fbank/*/"
run "data-mfcc.align.2/*"
run "data-phasediff/*/"
run "exp_pdnn/*/"
run "fbank/"
run "mfcc.align.2/"
run "phasediff/"
