
all:
KALDIBINDIR = /usr/local/kaldi/src/bin
EXTRA_CXXFLAGS = -Wno-sign-compare -I$(KALDIBINDIR)/../
include $(KALDIBINDIR)/../kaldi.mk

BINFILES = compute-phasediff-feats 

OBJFILES = 

TESTFILES =

ADDLIBS = $(KALDIBINDIR)/../feat/kaldi-feat.a $(KALDIBINDIR)/../transform/kaldi-transform.a $(KALDIBINDIR)/../gmm/kaldi-gmm.a \
         $(KALDIBINDIR)/../thread/kaldi-thread.a $(KALDIBINDIR)/../tree/kaldi-tree.a $(KALDIBINDIR)/../matrix/kaldi-matrix.a \
         $(KALDIBINDIR)/../util/kaldi-util.a $(KALDIBINDIR)/../base/kaldi-base.a

include $(KALDIBINDIR)/../makefiles/default_rules.mk

