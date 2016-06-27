

./compute-phasediff-feats --verbose=2 --config=../../conf/phasediff.conf \
            scp,p:wav.1.scp \
            scp,p:wav.3.scp \
            scp,p:wav.4.scp \
            scp,p:wav.5.scp \
            scp,p:wav.6.scp \
            ark,scp:out.ark,out.scp 

select-feats 0-255 scp:out.scp ark,t:- |head -2
select-feats 1536-1792 scp:out.scp ark,t:- |head -2
select-feats 3072-3327 scp:out.scp ark,t:- |head -2
select-feats 4608-4863 scp:out.scp ark,t:- |head -2
select-feats 6144-6399 scp:out.scp ark,t:- |head -2
