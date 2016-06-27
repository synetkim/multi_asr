#!/bin/bash


#if [ $# -ne 1 ]; then
#    echo "Usage: $0 <trusted-system>"
#    exit 1;
#fi

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
## This relates to the queue.
cmd=run.pl
nj=10

enhan=close
data=data
trusted_system=data-fbank

#for x in dt05_real_$enhan et05_real_$enhan dt05_simu_$enhan et05_simu_$enhan; do 
for x in tr05_real_$enhan tr05_simu_$enhan dt05_real_$enhan ; do 
    # 1. backup original $data
    cp $data/$x/feats.scp $data/$x/feats.scp.bk
    mkdir -p $data/$x/data_matchedframe

    # 2. split feats.scp for parallel compute
    split_scps=""
    trusted_split_scps=""
    for n in $(seq $nj); do
	split_scps="$split_scps $data/$x/data_matchedframe/feats.dsrtk.$n.scp"
	trusted_split_scps="$trusted_split_scps $data/$x/data_matchedframe/trusted_feats.$n.scp"
    done
    utils/split_scp.pl $data/$x/feats.scp $split_scps || exit 1;
    utils/split_scp.pl $trusted_system/${x}/feats.scp $trusted_split_scps || exit 1;

    # 3. add one frame at each utterance
    $cmd JOB=1:$nj exp/make_dsrtk/$x/make_dsrtk_match_frame.$x.JOB.log \
	./forKaldi/match-dsrtk-frame-length-by-utt scp:$data/$x/data_matchedframe/feats.dsrtk.JOB.scp \
	scp:$data/$x/data_matchedframe/trusted_feats.JOB.scp \
	ark,scp:$data/$x/data_matchedframe/matched.dsrtk.JOB.ark,$data/$x/data_matchedframe/matched.dsrtk.JOB.scp || exit 1;
    
    # Remove temp files
    rm $data/$x/data_matchedframe/trusted_feats.*.scp;
    
    # 4. combine to one feats.scp
    rm $data/$x/feats.scp
    for n in $(seq $nj); do
	cat $data/$x/data_matchedframe/matched.dsrtk.$n.scp || exit 1;
    done > $data/$x/feats.scp
    
    # 5. compute cmvn again
    steps/compute_cmvn_stats.sh $data/$x exp/make_dsrtk/$x $data/$x/data_matchedframe || exit 1;
done

#utils/combine_data.sh $data/dt05_multi_$enhan $data/dt05_simu_$enhan $data/dt05_real_$enhan

