// featbin/compute-spectrogram-feats.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "feat/feature-spectrogram.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create spectrogram feature files.\n"
        "Usage:  compute-phasediff-feats.cc [options...] w1 w3 w4 w5 w6 <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    SpectrogramOptions spec_opts;
    bool subtract_mean = false;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    // Define defaults for gobal options
    std::string output_format = "kaldi";

    // Register the option struct
    spec_opts.Register(&po);
    // Register the options
    po.Register("output-format", &output_format, "Format of the output files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each feature file [CMS]; not recommended to do it this way. ");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments to process (in seconds).");

    // OPTION PARSING ..........................................................
    //

    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      KALDI_LOG << "arg #: "<<po.NumArgs();
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier1 = po.GetArg(1);
    std::string wav_rspecifier3 = po.GetArg(2);
    std::string wav_rspecifier4 = po.GetArg(3);
    std::string wav_rspecifier5 = po.GetArg(4);
    std::string wav_rspecifier6 = po.GetArg(5);

    std::string output_wspecifier = po.GetArg(6);

    Spectrogram spec(spec_opts);

    SequentialTableReader<WaveHolder> reader1(wav_rspecifier1);
    SequentialTableReader<WaveHolder> reader3(wav_rspecifier3);
    SequentialTableReader<WaveHolder> reader4(wav_rspecifier4);
    SequentialTableReader<WaveHolder> reader5(wav_rspecifier5);
    SequentialTableReader<WaveHolder> reader6(wav_rspecifier6);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;

    if (output_format == "kaldi") {
      if (!kaldi_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else if (output_format == "htk") {
      if (!htk_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else {
      KALDI_ERR << "Invalid output_format string " << output_format;
    }

    int32 num_utts = 0, num_success = 0;
    for (; !reader1.Done(); reader1.Next(), reader3.Next(), reader4.Next(), reader5.Next(), reader6.Next()) {
      num_utts++;
      std::string utt = reader1.Key();
      const WaveData &wave_data1 = reader1.Value();
      const WaveData &wave_data3 = reader3.Value();
      const WaveData &wave_data4 = reader4.Value();
      const WaveData &wave_data5 = reader5.Value();
      const WaveData &wave_data6 = reader6.Value();
      if (wave_data1.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data1.Duration() << " sec): producing no output.";
        continue;
      }
      int32 this_chan = 0;

      if (spec_opts.frame_opts.samp_freq != wave_data1.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << spec_opts.frame_opts.samp_freq << " but data has "
                  << wave_data1.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;

      SubVector<BaseFloat> waveform1(wave_data1.Data(), this_chan);
      SubVector<BaseFloat> waveform3(wave_data3.Data(), this_chan);
      SubVector<BaseFloat> waveform4(wave_data4.Data(), this_chan);
      SubVector<BaseFloat> waveform5(wave_data5.Data(), this_chan);
      SubVector<BaseFloat> waveform6(wave_data6.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        spec.ComputeFFTwithPhaseDifference(waveform1, waveform3, waveform4, waveform5, waveform6, &features, NULL);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        continue;
      }
      if (output_format == "kaldi") {
        kaldi_writer.Write(utt, features);
      } else {
      }
      if(num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

