
echo "=================================================="
echo "./0.data.sh"
echo "=================================================="
#./steps_featext/0.data.org.sh 
#./steps_featext/0.data.beam.sh 
echo "=================================================="
echo "./1 feats"
echo "=================================================="
#./steps_featext/1.mfcc.sh || exit 1;
#./changeUttID.sh
#./steps_featext/1.6mic-mfcc.sh || exit 1;
echo "=================================================="
echo "./3 gmm"
echo "=================================================="
#./0.gmmalign.sh close || exit 1;
#./0.gmmalign.sh beamform || exit 1;
#./0.gmmalign.sh noisy || exit 1;
#./0.gmmalign.sh 6ch || exit 1;
echo "=================================================="
echo "./1 fmllr feature"
echo "=================================================="
#./steps_featext/1.fmllr.sh || exit 1;
#./steps_featext/1.fmllr-6ch.sh || exit 1;
echo "=================================================="
echo "./2.data_prep.sh"
echo "=================================================="
#./1.getAlignment.sh || exit 1;
echo "=================================================="
echo "./2.data_prep.sh"
echo "=================================================="
#./2.data_prep.kaldi.classification.sh close|| exit 1;
#./2.data_prep.kaldi.classification.sh beamform|| exit 1;
#./2.data_prep.kaldi.classification.sh noisy|| exit 1;
#./2.data_prep.kaldi.classification.sh 6ch|| exit 1;
echo "=================================================="
echo "./run-lstm.sh"
echo "=================================================="
#./run-lstm.class.sh noisy
echo "=================================================="
echo "finish"
echo "=================================================="
