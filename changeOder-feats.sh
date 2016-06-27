#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

#fbank 1000 + 40
# ch1ch3ch4ch5ch6, ..., ch1ch3ch4ch5ch6
# ch1x5frame, ch2x5frame, ..., ch5x5frame


echo "Start. change order"
working_dir=exp_pdnn
nj=4
testnj=4

$cmd JOB=1:$nj $working_dir/log/order_train.JOB.log \
 select-feats 0-39,200-239,400-439,600-639,800-839,40-79,240-279,440-479,640-679,840-879,80-119,280-319,480-519,680-719,880-919,120-159,320-359,520-559,720-759,920-959,160-199,360-399,560-599,760-799,960-999,1000-1039 \
 scp:exp_pdnn/data/train.JOB.scp ark,scp:exp_pdnn/data/train.order.JOB.ark,exp_pdnn/data/train.order.JOB.scp || exit 1;

$cmd JOB=1:$testnj $working_dir/log/order_valid.JOB.log \
 select-feats 0-39,200-239,400-439,600-639,800-839,40-79,240-279,440-479,640-679,840-879,80-119,280-319,480-519,680-719,880-919,120-159,320-359,520-559,720-759,920-959,160-199,360-399,560-599,760-799,960-999,1000-1039 \
 scp:exp_pdnn/data/valid.JOB.scp ark,scp:exp_pdnn/data/valid.order.JOB.ark,exp_pdnn/data/valid.order.JOB.scp || exit 1;


echo "Done. change order"
