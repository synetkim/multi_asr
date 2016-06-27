wav-copy scp:rh.scp ark,scp:orgh.ark,orgh.scp
wav-copy scp:rc1.scp ark,scp:orgc1.ark,orgc1.scp
wav-copy scp:rc3.scp ark,scp:orgc3.ark,orgc3.scp
wav-copy scp:rc4.scp ark,scp:orgc4.ark,orgc4.scp
wav-copy scp:rc5.scp ark,scp:orgc5.ark,orgc5.scp
wav-copy scp:rc6.scp ark,scp:orgc6.ark,orgc6.scp


./align-wav --verbose=2 scp:orgh.scp scp:orgc5.scp scp:orgc1.scp scp:orgc3.scp scp:orgc4.scp scp:orgc6.scp \
	ark,scp:newh.ark,newh.scp ark,scp:newc5.ark,newc5.scp ark,scp:newc1.ark,newc1.scp ark,scp:newc3.ark,newc3.scp \
	ark,scp:newc4.ark,newc4.scp ark,scp:newc6.ark,newc6.scp 

echo "==========================================="
wav-to-duration scp:orgh.scp ark,t:- |tail -3
wav-to-duration scp:newh.scp ark,t:- |tail -3
