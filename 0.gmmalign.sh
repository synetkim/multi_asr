#!/bin/bash

cmd=run.pl
. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. parse_options.sh || exit 1;

echo "=================================================="
echo "Run GMM and get alignment"
echo "=================================================="
chime3_data=/data-local/corpora/CHiME3/
enhancement_data=xx
enhancement_method=$1
enhan=$enhancement_method

local/run_gmm.newdata.sh $enhancement_method $enhancement_data data

echo "=================================================="
echo "finish"
echo "=================================================="
