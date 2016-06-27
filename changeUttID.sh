


function run(){
indir=$1
list=`find ./$indir -name feats.scp`
list1=`find ./$indir -name cmvn.scp`
list2=`find ./$indir -name text`
list3=`find ./$indir -name *utt*`
listsum="$list $list1 $list2 $list3"

for f in $listsum; do
	echo $f
	sed -i 's/\.CH[0-6]//g' $f
	sed -i 's/\._REAL/_REAL/g' $f
	sed -i 's/\._SIMU/_SIMU/g' $f
done
}

function run_wav(){
indir=$1
list=`find ./$indir -name wav.scp`
listsum="$list"

for f in $listsum; do
	echo $f
	sed -i 's/\.CH[0-6]_REAL/_REAL/g' $f
	sed -i 's/\.CH[0-6]_SIMU/_SIMU/g' $f
done
}

run data/
run_wav data/
