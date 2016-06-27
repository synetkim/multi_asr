// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "\n"
        "Usage:  nnet-forward [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    int32 time_shift = 0;
    po.Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames."); 

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    //nnet.Read(model_filename);
    // optionally remove softmax,
    //Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    //if (no_softmax) {
    //  if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
    //    KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
    //    nnet.RemoveComponent(nnet.NumComponents()-1);
    //  } else {
    //    KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
    //  }
    //}

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts); 

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    //nnet.SetDropoutRetention(1.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;


    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";

      
      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }

      // time-shift, copy the last frame of LSTM input N-times,
      if (time_shift > 0) {
        int32 last_row = mat.NumRows() - 1; // last row,
        mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
        for (int32 r = last_row+1; r<mat.NumRows(); r++) {
          mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
        }
      }
      
      // push it to gpu,
      feats = mat;

      // fwd-pass, feature transform,
      //nnet_transf.Feedforward(feats, &feats_transf);
      //if (!KALDI_ISFINITE(feats_transf.Sum())) { // check there's no nan/inf,
      //  KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      //}

      // fwd-pass, nnet,
      //nnet.Feedforward(feats_transf, &nnet_out);
      //if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
      //  KALDI_ERR << "NaN or inf found in nn-output for " << utt;
      //}
 
      nnet_out = feats;
      
      // convert posteriors to log-posteriors,
      if (apply_log) {
        if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
          KALDI_WARN << utt << " "
                     << "Applying 'log' to data which don't seem to be probabilities "
                     << "(is there a softmax somwhere?)";
        }
        nnet_out.Add(1e-20); // avoid log(0),
        nnet_out.ApplyLog();
      }
     
      // subtract log-priors from log-posteriors or pre-softmax,
      if (prior_opts.class_frame_counts != "") {
        if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
          KALDI_WARN << utt << " " 
                     << "Subtracting log-prior on 'probability-like' data in range [0..1] " 
                     << "(Did you forget --no-softmax=true or --apply-log=true ?)";
        }
        pdf_prior.SubtractOnLogpost(&nnet_out);
      }

      // download from GPU,
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);

      // time-shift, remove N first frames of LSTM output,
      if (time_shift > 0) {
        Matrix<BaseFloat> tmp(nnet_out_host);
        nnet_out_host = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
      }

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
