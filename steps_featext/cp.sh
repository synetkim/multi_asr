
sudo cp align-wav.cc /usr/local/kaldi/src/featbin/
sudo cp Makefile.featbin /usr/local/kaldi/src/featbin/Makefile
cd /usr/local/kaldi/src/featbin/
sudo make -j 48
cd -

#cd /usr/local/kaldi/src/feat/
#sudo cp feature-mfcc.cc /usr/local/kaldi/src/feat/
#sudo cp feature-mfcc.h /usr/local/kaldi/src/feat/
#sudo cp feature-spectrogram.h /usr/local/kaldi/src/feat/
#sudo cp feature-spectrogram.cc /usr/local/kaldi/src/feat/
#sudo cp feature-functions.cc /usr/local/kaldi/src/feat/
#sudo cp feature-functions.h /usr/local/kaldi/src/feat/
#sudo make -j 48
#cd -
