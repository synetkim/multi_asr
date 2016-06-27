

sudo cp feature-spectrogram.h /usr/local/kaldi/src/feat/
sudo cp feature-spectrogram.cc /usr/local/kaldi/src/feat/
sudo cp feature-functions.h /usr/local/kaldi/src/feat/
sudo cp feature-functions.cc /usr/local/kaldi/src/feat/
cd /usr/local/kaldi/src/feat/
sudo make -j 24
cd -
make
