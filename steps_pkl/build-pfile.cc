// bin/build-pfile-from-ali.cc

// Copyright 2013  Carnegie Mellon University (Author: Yajie Miao)
//                 Johns Hopkins University (Author: Daniel Povey)

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"

/** @brief Build pfiles for Neural Network training from alignment.
 * The pfiles contains both the data vectors and their corresponding
 * class/state labels (zero-based).
*/

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Build pfiles for neural network training from alignment.\n"
        "Usage:  build-pfile [options] <1: input feature-rspecifier> <2: output feature-rspecifier>\n"
        "<pfile-wspecifier>\n"
        "e.g.: \n"
        " build-pfile in:features out:features \n"
        " \"|pfile_create -i - -o pfile.1 -f 143 -l 1\" ";

    ParseOptions po(usage);

    int32 every_nth_frame = 1;
    po.Register("every-nth-frame", &every_nth_frame, "This option will cause it to print "
                "out only every n'th frame (for subsampling)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier1 = po.GetArg(1),
        feature_rspecifier2 = po.GetArg(2),
        pfile_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feature_reader1(feature_rspecifier1);
    SequentialBaseFloatMatrixReader feature_reader2(feature_rspecifier2);

    int32 num_done = 0;
    int32 num_utt = 0;

    KALDI_ASSERT(every_nth_frame >= 1);
    
    Output ko(pfile_wspecifier, false);
    int32 dim1 = 0;
    int32 dim2 = 0;

    for (; !feature_reader1.Done(); feature_reader1.Next()) {
      std::string key = feature_reader1.Key();
      //TODO: check key between feat1 and feat2 to be matched

      const Matrix<BaseFloat> &feats1 = feature_reader1.Value();
      const Matrix<BaseFloat> &feats2 = feature_reader2.Value();
      dim1 = feats1.NumCols();
      dim2 = feats2.NumCols();
	
      //KALDI_LOG << "BaseFloat size: " <<sizeof(BaseFloat);
      //KALDI_LOG << "input-frame number: " << feats1.NumRows();
      //KALDI_LOG << "output-frame number: " << feats2.NumRows();
      for (size_t i = 0; i < feats1.NumRows(); i++) {
          std::stringstream ss;
          // Output sentence number and frame number
          ss << num_utt;
          ss << " ";
          ss << i ;
          // Output input-feature vector
          for (int32 d = 0; d < dim1; ++d) {
            ss << " ";
            ss << feats1(i, d);
          }
          // Output output-feature vector
          for (int32 d = 0; d < dim2; ++d) {
            ss << " ";
            ss << feats2(i, d);
          }
          ko.Stream() << ss.str().c_str();
          ko.Stream() << "\n";
	  num_done ++;
      }
      num_utt ++;
      feature_reader2.Next();
    }
    ko.Close();
    KALDI_LOG << "Utterance number: " << num_utt ;
    KALDI_LOG << "Number of Done " << num_done ;
    KALDI_LOG << dim1 << " input feat dim; "
              << dim2 << " output feat dim.";
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


