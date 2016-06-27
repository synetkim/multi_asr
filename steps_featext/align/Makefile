
all:
KALDIBINDIR = /usr/local/kaldi/src/featbin
EXTRA_CXXFLAGS = -Wno-sign-compare -I$(KALDIBINDIR)/../
include $(KALDIBINDIR)/../kaldi.mk

BINFILES = align-wav # match-pncc-frame-length #build-pfile-from-ali-noise-label-auto build-pfile-from-ali-noise-label

TESTFILES =

OBJFILES =

LIBNAME =

ADDLIBS = $(KALDIBINDIR)/../feat/kaldi-feat.a $(KALDIBINDIR)/../transform/kaldi-transform.a $(KALDIBINDIR)/../gmm/kaldi-gmm.a \
         $(KALDIBINDIR)/../thread/kaldi-thread.a $(KALDIBINDIR)/../tree/kaldi-tree.a $(KALDIBINDIR)/../matrix/kaldi-matrix.a \
         $(KALDIBINDIR)/../util/kaldi-util.a $(KALDIBINDIR)/../base/kaldi-base.a

TESTFILES =

include $(KALDIBINDIR)/../makefiles/default_rules.mk

