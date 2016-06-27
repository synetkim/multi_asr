
echo "=================================================="
echo "./0.data.sh"
echo "=================================================="
#./steps_featext/0.data.sh|| exit 1;
echo "=================================================="
echo "./1.feature.sh"
echo "=================================================="
#./steps_featext/1.mfcc-close.sh || exit 1;
echo "=================================================="
echo "./run.sh"
echo "=================================================="
. ./path.sh
. ./cmd.sh
enhancement_method=close
enhancement_data=xx
#local/run_gmm.newdata.real.sh $enhancement_method $enhancement_data data
#./1.getAlignment.sh $enhancement_method || exit 1;
#./2.data_prep.kaldi.classification.sh || exit 1;
./run-dnn.class.sh $enhancement_method || exit 1;
echo "=================================================="
echo "finish"
echo "=================================================="
