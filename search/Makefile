# SPDX-License-Identifier: Apache-2.0
# make cpu : build/{prof,rank,sa,weakkeys}_cpu with the CPU metric, build/check, build/state and build/sactl
# make gpu : build/{prof,rank,sa,weakkeys}_gpu with the HIP metric (HIPCC, HIP_ARCH), and build/sactl
# The tools are C++17; melty2_init.c of the library is compiled as C.
CC ?= cc
CXX ?= c++
CFLAGS ?= -O3 -march=native
CXXFLAGS ?= -O3 -march=native
HIPCC ?= hipcc
HIP_ARCH ?= gfx1030
B := build
ROOT := ..
INC := -I. -I$(ROOT)/include
TOOLS := prof rank sa weakkeys
HEADERS := arx.hpp metric.hpp keymod.hpp pidfile.hpp

.PHONY: cpu gpu clean
cpu: $(TOOLS:%=$(B)/%_cpu) $(B)/check $(B)/state $(B)/sactl
gpu: $(TOOLS:%=$(B)/%_gpu) $(B)/sactl

$(B)/%.o: %.cc $(HEADERS) | $(B)
	$(CXX) -std=c++17 $(CXXFLAGS) -Wall -Wextra $(INC) -c $< -o $@
$(B)/melty2_init.o: $(ROOT)/src/melty2_init.c | $(B)
	$(CC) -std=c99 $(CFLAGS) $(INC) -c $< -o $@
$(B)/metric_gpu.o: metric_gpu.hip arx.hpp metric.hpp | $(B)
	$(HIPCC) -std=c++17 --offload-arch=$(HIP_ARCH) -O3 -Wall $(INC) -c $< -o $@

COMMON := $(B)/metric_common.o $(B)/melty2_init.o
$(B)/%_cpu: $(B)/%.o $(B)/metric_cpu.o $(COMMON)
	$(CXX) $^ -o $@
$(B)/%_gpu: $(B)/%.o $(B)/metric_gpu.o $(COMMON)
	$(HIPCC) --offload-arch=$(HIP_ARCH) $^ -o $@

# make stest : build/testu01_stest, the tests of BigCrush (but five) on GPU for the ARX structures, by the
# library testu01_gpu (../testu01_gpu, built there with HIPCC, HIP_ARCH and TESTU01), which links the
# probdist library of TestU01 (TESTU01)
TESTU01 ?= /usr/local
TU01G := $(ROOT)/testu01_gpu
.PHONY: stest $(TU01G)/build/libtestu01_gpu.a
stest: $(B)/testu01_stest
$(TU01G)/build/libtestu01_gpu.a:
	$(MAKE) -C $(TU01G) HIPCC=$(HIPCC) HIP_ARCH=$(HIP_ARCH) TESTU01=$(TESTU01) HOST_ARCH="$(HOST_ARCH)" build/libtestu01_gpu.a
$(B)/testu01_stest.o: testu01_stest.hip $(TU01G)/include/testu01_gpu.hpp $(TU01G)/include/testu01_gpu_params.hpp arx.hpp keymod.hpp pidfile.hpp | $(B)
	$(HIPCC) -std=c++17 --offload-arch=$(HIP_ARCH) -O3 -Wall $(INC) -I$(TU01G)/include -c $< -o $@
$(B)/testu01_stest: $(B)/testu01_stest.o $(B)/melty2_init.o $(TU01G)/build/libtestu01_gpu.a
	$(HIPCC) --offload-arch=$(HIP_ARCH) $^ -L$(TESTU01)/lib -Wl,-rpath,$(TESTU01)/lib -lprobdist -lmylib -lhiprtc -lrocfft -o $@

$(B)/check: check.cc arx.hpp | $(B)
	$(CXX) -std=c++17 $(CXXFLAGS) -Wall -Wextra $(INC) $< -o $@
$(B)/state: state.cc $(B)/melty2_init.o arx.hpp keymod.hpp | $(B)
	$(CXX) -std=c++17 $(CXXFLAGS) -Wall -Wextra $(INC) -pthread state.cc $(B)/melty2_init.o -o $@
$(B)/sactl: sactl.cc | $(B)
	$(CXX) -std=c++17 $(CXXFLAGS) -Wall -Wextra $< -o $@

$(B):
	mkdir -p $@
clean:
	rm -rf $(B)
