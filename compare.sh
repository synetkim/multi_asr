#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

feat=data-fbank
feat=data

for set in tr05 dt05 et05; do
rm -rf $feat/*/split* 
sed -i 's/CH0//g' $feat/${set}_real_close/feats.scp
sed -i 's/CH0//g' $feat/${set}_real_close/spk2utt
sed -i 's/CH0//g' $feat/${set}_real_close/text
sed -i 's/CH0//g' $feat/${set}_real_close/utt2spk
sed -i 's/CH0//g' $feat/${set}_real_close/wav.scp
done
train_outfeats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$feat/tr05_real_close/utt2spk scp:$feat/tr05_real_close/cmvn.scp scp:$feat/tr05_real_close/feats.scp ark:- |"
dev_outfeats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$feat/dt05_real_close/utt2spk scp:$feat/dt05_real_close/cmvn.scp scp:$feat/dt05_real_close/feats.scp ark:- |"
eval_outfeats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$feat/et05_real_close/utt2spk scp:$feat/et05_real_close/cmvn.scp scp:$feat/et05_real_close/feats.scp ark:- |"


echo "======================== compare Train ====================="
compare-feats "$train_outfeats" scp:exp_pdnn/data-new/tr05_real_dnnenhan/feats.scp
echo "======================== compare Dev ====================="
compare-feats "$dev_outfeats" scp:exp_pdnn/data-new/dt05_real_dnnenhan/feats.scp
echo "======================== compare Eval ====================="
compare-feats "$eval_outfeats" scp:exp_pdnn/data-new/et05_real_dnnenhan/feats.scp

select-feats 0-5 "$eval_outfeats" ark,t:-|head -5
select-feats 0-5 scp:exp_pdnn/data-new/et05_real_dnnenhan/feats.scp ark,t:-|head -5

