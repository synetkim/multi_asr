

copy-feats scp:pncc.scp ark,t:- |head -5
copy-feats scp:pncc.scp ark,t:- |tail -5

copy-feats scp:pncc.scp ark,t:- |wc -l


#./match-pncc-frame-length scp:pncc.scp ark,t:- |head -8
#./match-pncc-frame-length scp:pncc.scp ark,t:- |tail -8
#./match-pncc-frame-length scp:pncc.scp ark,scp:newpncc.ark,newpncc.scp

./match-dsrtk-frame-length scp:pncc.scp ark,t:- |head -8
./match-dsrtk-frame-length scp:pncc.scp ark,t:- |tail -8
./match-dsrtk-frame-length scp:pncc.scp ark,scp:newpncc.ark,newpncc.scp
