// featbin/align-wav.cc

// Copyright 2013-2014  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Copy archives of wave files\n"
        "\n"
        "Usage:  align-wav [options...] <wav-rspecifier> <wav-rspecifier>\n"
        "e.g. align-wav scp:headset.scp scp:5ch scp:1ch scp:3ch scp:4ch scp:6ch ark:headset ark:5ch ark:1ch ark:3ch ark:4ch ark:6ch\n"
        "See also: wav-to-duration extract-segments\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 12) {
      po.PrintUsage();
      exit(1);
    }

    std::string 
		wav_rspecifier_h1 = po.GetArg(1),
        wav_wspecifier_h1 = po.GetArg(7),
		wav_rspecifier_c5 = po.GetArg(2),
        wav_wspecifier_c5 = po.GetArg(8),
		wav_rspecifier_c1 = po.GetArg(3),
        wav_wspecifier_c1 = po.GetArg(9),
		wav_rspecifier_c3 = po.GetArg(4),
        wav_wspecifier_c3 = po.GetArg(10),
		wav_rspecifier_c4 = po.GetArg(5),
        wav_wspecifier_c4 = po.GetArg(11),
		wav_rspecifier_c6 = po.GetArg(6),
        wav_wspecifier_c6 = po.GetArg(12);

    int32 num_done = 0;

	///////////////////////////////////////
	// 1. find delay bet. headset vs. ch5   
	///////////////////////////////////////
    SequentialTableReader<WaveHolder> wav_reader_h1(wav_rspecifier_h1);
    TableWriter<WaveHolder> wav_writer_h1(wav_wspecifier_h1);
    SequentialTableReader<WaveHolder> wav_reader_c5(wav_rspecifier_c5);
    TableWriter<WaveHolder> wav_writer_c5(wav_wspecifier_c5);
    SequentialTableReader<WaveHolder> wav_reader_c1(wav_rspecifier_c1);
    TableWriter<WaveHolder> wav_writer_c1(wav_wspecifier_c1);
    SequentialTableReader<WaveHolder> wav_reader_c3(wav_rspecifier_c3);
    TableWriter<WaveHolder> wav_writer_c3(wav_wspecifier_c3);
    SequentialTableReader<WaveHolder> wav_reader_c4(wav_rspecifier_c4);
    TableWriter<WaveHolder> wav_writer_c4(wav_wspecifier_c4);
    SequentialTableReader<WaveHolder> wav_reader_c6(wav_rspecifier_c6);
    TableWriter<WaveHolder> wav_writer_c6(wav_wspecifier_c6);

	///////////////////////////////////////
	// 2. change wav files    
	///////////////////////////////////////
    for (; !wav_reader_h1.Done(); wav_reader_h1.Next(),wav_reader_c5.Next(),wav_reader_c1.Next(),wav_reader_c3.Next(),wav_reader_c4.Next(),wav_reader_c6.Next()) {
	  const WaveData &wave_data_h1 = wav_reader_h1.Value();
	  const WaveData &wave_data_c5 = wav_reader_c5.Value();
	  const WaveData &wave_data_c1 = wav_reader_c1.Value();
	  const WaveData &wave_data_c3 = wav_reader_c3.Value();
	  const WaveData &wave_data_c4 = wav_reader_c4.Value();
	  const WaveData &wave_data_c6 = wav_reader_c6.Value();

	  SubVector<BaseFloat> x(wave_data_h1.Data(), 0);
	  SubVector<BaseFloat> tmpy(wave_data_c5.Data(), 0);

      int delay = -1;
      int finaldelay = -1;
      double finalr = -1; 
      int maxdelay = 3000;
	  int i,j;
	  double mx,my,sx,sy,sxy,denom,r;
	  int n=x.Dim();

	  int truncatedSample = 0;
	  if (x.Dim() < tmpy.Dim()) {
		truncatedSample = ( tmpy.Dim() - x.Dim() )/2;
	  }
	  else if (x.Dim() > tmpy.Dim()){
		return -1;
      }
	  SubVector<BaseFloat> y(&tmpy(truncatedSample), x.Dim());

	  mx = 0;my = 0;   
	  for (i=0;i<n;i++) {
		  mx += x(i);my += y(i);
	  }
	  mx /= n;my /= n;
	  sx = 0;sy = 0;
	  for (i=0;i<n;i++) {
		  sx += (x(i) - mx) * (x(i) - mx);sy += (y(i) - my) * (y(i) - my);
	  }
	  denom = sqrt(sx*sy);

	  for (delay=-maxdelay;delay<maxdelay;delay++) {
		  sxy = 0;
		  for (i=0;i<n;i++) {
			  j = i + delay;
			  if (j < 0 || j >= n) { 
				  continue;
			  }
			  else {
				  sxy += (x(i) - mx) * (y(j) - my);
			  }
		  }
		  r = sxy / denom;
		  if (finalr < fabs(r)) {
			finalr=fabs(r);
			finaldelay=delay;
		  }
	  }
      // if delay > 0, remove close sample
	  if (finaldelay < 0) {
		finaldelay = abs(finaldelay);

		const Matrix<BaseFloat> &mat_h1 = wave_data_h1.Data();
		SubMatrix<BaseFloat> segment_matrix_h1(mat_h1, 0, 1, finaldelay, n-finaldelay);
		WaveData segment_wave_h1(wave_data_h1.SampFreq(), segment_matrix_h1);
        wav_writer_h1.Write(wav_reader_h1.Key(), segment_wave_h1);

		const Matrix<BaseFloat> &mat_c5 = wave_data_c5.Data();
		SubMatrix<BaseFloat> segment_matrix_c5(mat_c5, 0, 1, truncatedSample, n-finaldelay);
		WaveData segment_wave_c5(wave_data_c5.SampFreq(), segment_matrix_c5);
        wav_writer_c5.Write(wav_reader_c5.Key(), segment_wave_c5);

		const Matrix<BaseFloat> &mat_c1 = wave_data_c1.Data();
		SubMatrix<BaseFloat> segment_matrix_c1(mat_c1, 0, 1, truncatedSample, n-finaldelay);
		WaveData segment_wave_c1(wave_data_c1.SampFreq(), segment_matrix_c1);
        wav_writer_c1.Write(wav_reader_c1.Key(), segment_wave_c1);

		const Matrix<BaseFloat> &mat_c3 = wave_data_c3.Data();
		SubMatrix<BaseFloat> segment_matrix_c3(mat_c3, 0, 1, truncatedSample, n-finaldelay);
		WaveData segment_wave_c3(wave_data_c3.SampFreq(), segment_matrix_c3);
        wav_writer_c3.Write(wav_reader_c3.Key(), segment_wave_c3);

		const Matrix<BaseFloat> &mat_c4 = wave_data_c4.Data();
		SubMatrix<BaseFloat> segment_matrix_c4(mat_c4, 0, 1, truncatedSample, n-finaldelay);
		WaveData segment_wave_c4(wave_data_c4.SampFreq(), segment_matrix_c4);
        wav_writer_c4.Write(wav_reader_c4.Key(), segment_wave_c4);

		const Matrix<BaseFloat> &mat_c6 = wave_data_c6.Data();
		SubMatrix<BaseFloat> segment_matrix_c6(mat_c6, 0, 1, truncatedSample, n-finaldelay);
		WaveData segment_wave_c6(wave_data_c6.SampFreq(), segment_matrix_c6);
        wav_writer_c6.Write(wav_reader_c6.Key(), segment_wave_c6);
	  }
      // if delay < 0, remove ch1,3,4,5,6 sample
	  else if (finaldelay >= 0) {
		const Matrix<BaseFloat> &mat_h1 = wave_data_h1.Data();
		SubMatrix<BaseFloat> segment_matrix_h1(mat_h1, 0, 1, 0, n-finaldelay);
		WaveData segment_wave_h1(wave_data_h1.SampFreq(), segment_matrix_h1);
        wav_writer_h1.Write(wav_reader_h1.Key(), segment_wave_h1);

		const Matrix<BaseFloat> &mat_c5 = wave_data_c5.Data();
		SubMatrix<BaseFloat> segment_matrix_c5(mat_c5, 0, 1, finaldelay+truncatedSample, n-finaldelay);
		WaveData segment_wave_c5(wave_data_c5.SampFreq(), segment_matrix_c5);
        wav_writer_c5.Write(wav_reader_c5.Key(), segment_wave_c5);

		const Matrix<BaseFloat> &mat_c1 = wave_data_c1.Data();
		SubMatrix<BaseFloat> segment_matrix_c1(mat_c1, 0, 1, finaldelay+truncatedSample, n-finaldelay);
		WaveData segment_wave_c1(wave_data_c1.SampFreq(), segment_matrix_c1);
        wav_writer_c1.Write(wav_reader_c1.Key(), segment_wave_c1);

		const Matrix<BaseFloat> &mat_c3 = wave_data_c3.Data();
		SubMatrix<BaseFloat> segment_matrix_c3(mat_c3, 0, 1, finaldelay+truncatedSample, n-finaldelay);
		WaveData segment_wave_c3(wave_data_c3.SampFreq(), segment_matrix_c3);
        wav_writer_c3.Write(wav_reader_c3.Key(), segment_wave_c3);

		const Matrix<BaseFloat> &mat_c4 = wave_data_c4.Data();
		SubMatrix<BaseFloat> segment_matrix_c4(mat_c4, 0, 1, finaldelay+truncatedSample, n-finaldelay);
		WaveData segment_wave_c4(wave_data_c4.SampFreq(), segment_matrix_c4);
        wav_writer_c4.Write(wav_reader_c4.Key(), segment_wave_c4);

		const Matrix<BaseFloat> &mat_c6 = wave_data_c6.Data();
		SubMatrix<BaseFloat> segment_matrix_c6(mat_c6, 0, 1, finaldelay+truncatedSample, n-finaldelay);
		WaveData segment_wave_c6(wave_data_c6.SampFreq(), segment_matrix_c6);
        wav_writer_c6.Write(wav_reader_c6.Key(), segment_wave_c6);
      }

      KALDI_LOG << "final r:  " << finalr << " at final delay: " << finaldelay<< " truncatedSample:  " << truncatedSample <<" final length: "<<n-finaldelay;
      num_done++;
    }
    KALDI_LOG << "Copied " << num_done << " wave files";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

