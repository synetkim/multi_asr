
all:
#KALDIBINDIR = /nobackupp8/achandra/Workspace/Software/kaldi/src/bin
KALDIBINDIR = /usr/local/kaldi/src/bin
EXTRA_CXXFLAGS = -Wno-sign-compare -I$(KALDIBINDIR)/../
include $(KALDIBINDIR)/../kaldi.mk

BINFILES = match-dsrtk-frame-length \
	match-dsrtk-frame-length-by-utt \
	match-pncc-frame-length \
	build-pfile-from-ali-noise-label-auto \
	build-pfile-from-ali-noise-label

TESTFILES =

OBJFILES =

LIBNAME =

ADDLIBS = $(KALDIBINDIR)/../lm/kaldi-lm.a $(KALDIBINDIR)/../decoder/kaldi-decoder.a $(KALDIBINDIR)/../lat/kaldi-lat.a \
          $(KALDIBINDIR)/../hmm/kaldi-hmm.a $(KALDIBINDIR)/../transform/kaldi-transform.a $(KALDIBINDIR)/../gmm/kaldi-gmm.a \
          $(KALDIBINDIR)/../tree/kaldi-tree.a $(KALDIBINDIR)/../matrix/kaldi-matrix.a  $(KALDIBINDIR)/../util/kaldi-util.a \
          $(KALDIBINDIR)/../base/kaldi-base.a  $(KALDIBINDIR)/../thread/kaldi-thread.a

TESTFILES =

include $(KALDIBINDIR)/../makefiles/default_rules.mk

