
echo "=================================================="
echo "./0.data.sh"
echo "=================================================="
#./steps_featext/0.data.org.sh|| exit 1;
#./steps_featext/align/alignHeadsetDistantMic.sh || exit 1;
./steps_featext/1.mfcc-all.align.2.sh || exit 1;
echo "=================================================="
echo "./1 distant multi-mic (including et05)"
echo "=================================================="
./steps_featext/1.6mic.sh data-mfcc.align.2 fft || exit 1;
./steps_featext/1.6mic.sh data-mfcc.align.2 fbank || exit 1;
echo "=================================================="
echo "./1 close mic"
echo "=================================================="
./steps_featext/1.close.sh data-mfcc.align.2 fft|| exit 1;
./steps_featext/1.close.sh data-mfcc.align.2 fbank|| exit 1;
echo "=================================================="
echo "./2.data_prep.sh"
echo "=================================================="
#./2.data_prep.kaldi.sh|| exit 1;
echo "=================================================="
echo "./run-dnn.sh"
echo "=================================================="
#./run-lstm.sh|| exit 1;
#./run-cnn.sh|| exit 1;
#./run-dnn.sh|| exit 1;
echo "=================================================="
echo "./run.sh"
echo "=================================================="
#./run.dnnenhan.real.sh || exit 1;
echo "=================================================="
echo "finish"
echo "=================================================="
