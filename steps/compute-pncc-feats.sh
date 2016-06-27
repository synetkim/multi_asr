#!/bin/bash

pnccpath=/data-local/sykim/tool/PNCC_C
outdir=$2

#echo $2
#echo $pnccpath

cat $1 | while read line; do
  arr=($line)
  infile=${arr[${#arr[@]} - 1]}
  outfile=${arr[0]}
  #echo $infile
  #echo $outfile
  $pnccpath/wav2pncc -i $infile -o $outdir/$outfile -f $pnccpath/GTFB_40_1024_100_6800_16000.bin
done

