// featbin/copy-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy features [and possibly change format]\n"
        "Usage: copy-feats [options] <feature-rspecifier> <feature-wspecifier>\n"
        "or:   copy-feats [options] <feats-rxfilename> <feats-wxfilename>\n"
        "e.g.: copy-feats ark:- ark,scp:foo.ark,foo.scp\n"
        " or: copy-feats ark:foo.ark ark,t:txt.ark\n"
        "See also: copy-matrix, copy-feats-to-htk, copy-feats-to-sphinx, select-feats,\n"
        "extract-rows, subset-feats, subsample-feats, splice-feats, append-feats\n";

    ParseOptions po(usage);
    bool binary = true;
    bool htk_in = false;
    bool sphinx_in = false;
    bool compress = false;
    po.Register("htk-in", &htk_in, "Read input as HTK features");
    po.Register("sphinx-in", &sphinx_in, "Read input as Sphinx features");
    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;
    
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) 
    {
      // Copying tables of features.
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);

      if (!compress) 
      {
        KALDI_LOG << "!compress " ;
        BaseFloatMatrixWriter kaldi_writer(wspecifier);
        SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) 
          {
            Matrix<BaseFloat> orgFeatMat(kaldi_reader.Value());
            Matrix<BaseFloat> newFeatMat;
            int32 nRows=orgFeatMat.NumRows();
            int32 nCols=orgFeatMat.NumCols();

            //KALDI_LOG << "reader " <<kaldi_reader.Key();
            //KALDI_LOG << "reader " <<nRows;
            //KALDI_LOG << "reader " <<nCols;
            newFeatMat.Resize(nRows+7, nCols);
            newFeatMat.Range(0, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(1, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(2, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(3, nRows, 0, nCols).CopyFromMat(orgFeatMat.Range(0, nRows, 0, nCols));
            //KALDI_LOG << orgFeatMat.Range(nRows-1, 1, 0, nCols);
            newFeatMat.Range(nRows+3, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+4, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+5, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+6, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            //KALDI_LOG << "reader " <<newFeatMat.NumRows();
            kaldi_writer.Write(kaldi_reader.Key(), newFeatMat);
          }
      } 
      else 
      {
        CompressedMatrixWriter kaldi_writer(wspecifier);
        SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
        for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++)
        {
            Matrix<BaseFloat> orgFeatMat(kaldi_reader.Value());
            Matrix<BaseFloat> newFeatMat;
            int32 nRows=orgFeatMat.NumRows();
            int32 nCols=orgFeatMat.NumCols();
            newFeatMat.Resize(nRows+7, nCols);
            newFeatMat.Range(0, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(1, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(2, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(0, 1, 0, nCols));
            newFeatMat.Range(3, nRows, 0, nCols).CopyFromMat(orgFeatMat.Range(0, nRows, 0, nCols));
            newFeatMat.Range(nRows+3, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+4, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+5, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            newFeatMat.Range(nRows+6, 1, 0, nCols).CopyFromMat(orgFeatMat.Range(nRows-1, 1, 0, nCols));
            kaldi_writer.Write(kaldi_reader.Key(),
                             CompressedMatrix( newFeatMat));
        }
      }
      KALDI_LOG << "Copied " << num_done << " feature matrices.";
      return (num_done != 0 ? 0 : 1);
    } 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


