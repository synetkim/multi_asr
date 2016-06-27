// feat/feature-spectrogram.cc

// Copyright 2009-2012  Karel Vesely
// Copyright 2012  Navdeep Jaitly

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


#include "feat/feature-spectrogram.h"


namespace kaldi {

Spectrogram::Spectrogram(const SpectrogramOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Spectrogram::~Spectrogram() {
  if (srfft_ != NULL)
    delete srfft_;
}

void Spectrogram::ComputeFFTwithCoeff(const VectorBase<BaseFloat> &wave,
                          Matrix<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out =  opts_.frame_opts.PaddedWindowSize();
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out, cols_out);

  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  BaseFloat log_energy;

  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &window, (opts_.raw_energy ? &log_energy : NULL));

    // Compute energy after window function (not the raw one)
    if (!opts_.raw_energy)
      log_energy = log(std::max(VecVec(window, window),
                                std::numeric_limits<BaseFloat>::min()));

    if (srfft_ != NULL)  {// Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
      //KALDI_LOG << "Compute FFT using split-radix algorithm";
    }
    else{  // An alternative algorithm that works for non-powers-of-two
      RealFft(&window, true);
      KALDI_LOG << "An alternative algorithm that works for non-powers-of-two";
    }
    SubVector<BaseFloat> fft_coeff(window, 0, window.Dim());

    fft_coeff.ApplyFloor(std::numeric_limits<BaseFloat>::min());
    fft_coeff.ApplyLog();

    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    this_output.CopyFromVec(fft_coeff);
    if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
    }
    this_output(0) = log_energy;
  }
}

void Spectrogram::ComputeFFTwithPhaseDifference(const VectorBase<BaseFloat> &wave1,
						  const VectorBase<BaseFloat> &wave3,
						  const VectorBase<BaseFloat> &wave4,
						  const VectorBase<BaseFloat> &wave5,
						  const VectorBase<BaseFloat> &wave6,
                          Matrix<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave1.Dim(), opts_.frame_opts);
  int32 final_cols_out =  opts_.frame_opts.PaddedWindowSize()/2*5*5;
  int32 mid_cols_out =  opts_.frame_opts.PaddedWindowSize()/2*5;
  int32 start_cols_out =  opts_.frame_opts.PaddedWindowSize()/2;
  KALDI_LOG << "final_cols_out: "<< final_cols_out;
  KALDI_LOG << "mid_cols_out: "<< mid_cols_out;
  KALDI_LOG << "start_cols_out: "<< start_cols_out;
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave1.Dim() << ")";

  output->Resize(rows_out, final_cols_out);

  // Buffers
  Vector<BaseFloat> window1;  // windowed waveform.
  Vector<BaseFloat> window3;  // windowed waveform.
  Vector<BaseFloat> window4;  // windowed waveform.
  Vector<BaseFloat> window5;  // windowed waveform.
  Vector<BaseFloat> window6;  // windowed waveform.
  BaseFloat log_energy1;
  BaseFloat log_energy3;
  BaseFloat log_energy4;
  BaseFloat log_energy5;
  BaseFloat log_energy6;

  KALDI_LOG << "compute all frames: "<< rows_out;
  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    KALDI_LOG << "fram # "<< r;
    // Cut the window, apply window function
    ExtractWindow(wave1, r, opts_.frame_opts, feature_window_function_,
                  &window1, (opts_.raw_energy ? &log_energy1 : NULL));
    ExtractWindow(wave3, r, opts_.frame_opts, feature_window_function_,
                  &window3, (opts_.raw_energy ? &log_energy3 : NULL));
    ExtractWindow(wave4, r, opts_.frame_opts, feature_window_function_,
                  &window4, (opts_.raw_energy ? &log_energy4 : NULL));
    ExtractWindow(wave5, r, opts_.frame_opts, feature_window_function_,
                  &window5, (opts_.raw_energy ? &log_energy5 : NULL));
    ExtractWindow(wave6, r, opts_.frame_opts, feature_window_function_,
                  &window6, (opts_.raw_energy ? &log_energy6 : NULL));

    if (srfft_ != NULL)  {// Compute FFT using split-radix algorithm.
      srfft_->Compute(window1.Data(), true);
      srfft_->Compute(window3.Data(), true);
      srfft_->Compute(window4.Data(), true);
      srfft_->Compute(window5.Data(), true);
      srfft_->Compute(window6.Data(), true);
    }

    KALDI_LOG << "Extract fft coeff # "<< r;
    // FFT = > Phase Differences.
    Vector<BaseFloat> final_feature(final_cols_out);
    Vector<BaseFloat> mid_final_feature(mid_cols_out);

    KALDI_LOG << "window: "<< window1.Dim();
    ComputePowerSpectrumPhaseDifference(&window1, &window1,&window3,&window4,&window5,&window6,&mid_final_feature);
    final_feature.Range(mid_cols_out*0, mid_cols_out).CopyFromVec(mid_final_feature);
    KALDI_LOG << "done: "<<mid_cols_out*0<<" - "<< mid_cols_out*0+mid_cols_out;

    ComputePowerSpectrumPhaseDifference(&window3, &window1,&window3,&window4,&window5,&window6,&mid_final_feature);
    final_feature.Range(mid_cols_out*1, mid_cols_out).CopyFromVec(mid_final_feature);
    KALDI_LOG << "done: "<<mid_cols_out*1<<" - "<< mid_cols_out*1+mid_cols_out;
    
    ComputePowerSpectrumPhaseDifference(&window4, &window1,&window3,&window4,&window5,&window6,&mid_final_feature);
    final_feature.Range(mid_cols_out*2, mid_cols_out).CopyFromVec(mid_final_feature);
    KALDI_LOG << "done: "<<mid_cols_out*2<<" - "<< mid_cols_out*2+mid_cols_out;
    
    ComputePowerSpectrumPhaseDifference(&window5, &window1,&window3,&window4,&window5,&window6,&mid_final_feature);
    final_feature.Range(mid_cols_out*3, mid_cols_out).CopyFromVec(mid_final_feature);
    KALDI_LOG << "done: "<<mid_cols_out*3<<" - "<< mid_cols_out*3+mid_cols_out;
    
    ComputePowerSpectrumPhaseDifference(&window6, &window1,&window3,&window4,&window5,&window6,&mid_final_feature);
    final_feature.Range(mid_cols_out*4, mid_cols_out).CopyFromVec(mid_final_feature);
    KALDI_LOG << "done: "<<mid_cols_out*4<<" - "<< mid_cols_out*4+mid_cols_out;

    KALDI_LOG << "copy vectors "<< r;
    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    this_output.CopyFromVec(final_feature);
    KALDI_LOG << "Done... "<< r;

  }
}

void Spectrogram::ComputeFFTwithPhase(const VectorBase<BaseFloat> &wave,
                          Matrix<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out =  opts_.frame_opts.PaddedWindowSize();
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out, cols_out);

  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  BaseFloat log_energy;

  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &window, (opts_.raw_energy ? &log_energy : NULL));

    // Compute energy after window function (not the raw one)
    if (!opts_.raw_energy)
      log_energy = log(std::max(VecVec(window, window),
                                std::numeric_limits<BaseFloat>::min()));

    if (srfft_ != NULL)  {// Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
      //KALDI_LOG << "Compute FFT using split-radix algorithm";
    }
    else{  // An alternative algorithm that works for non-powers-of-two
      RealFft(&window, true);
      KALDI_LOG << "An alternative algorithm that works for non-powers-of-two";
    }
   // Convert the FFT into a power spectrum.
    ComputePowerSpectrumPhase(&window);
    
    SubVector<BaseFloat> magVec(window, 0, window.Dim()/2 + 1);
    Vector<BaseFloat> phaseVec(window.Dim()/2 - 1);
    phaseVec.CopyFromVec(window.Range(window.Dim()/2 + 1, window.Dim()/2 - 1));

    magVec.ApplyFloor(std::numeric_limits<BaseFloat>::min());
    magVec.ApplyLog();
  
    Vector<BaseFloat> final_feature(cols_out);
    final_feature.Range(0, window.Dim()/2 + 1).CopyFromVec(magVec);
    final_feature.Range(window.Dim()/2 + 1, window.Dim()/2 - 1).CopyFromVec(phaseVec);

    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    this_output.CopyFromVec(final_feature);
    if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
    }
    this_output(0) = log_energy;
  }
}


void Spectrogram::Compute(const VectorBase<BaseFloat> &wave,
                          Matrix<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out =  opts_.frame_opts.PaddedWindowSize()/2 +1;
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out, cols_out);

  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  BaseFloat log_energy;

  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &window, (opts_.raw_energy ? &log_energy : NULL));

    // Compute energy after window function (not the raw one)
    if (!opts_.raw_energy)
      log_energy = log(std::max(VecVec(window, window),
                                std::numeric_limits<BaseFloat>::min()));
    
    if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
    else  // An alternative algorithm that works for non-powers-of-two
      RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    power_spectrum.ApplyFloor(std::numeric_limits<BaseFloat>::min());
    power_spectrum.ApplyLog();

    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    this_output.CopyFromVec(power_spectrum);
    if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
    }
    this_output(0) = log_energy;
  }
}

}  // namespace kaldi
