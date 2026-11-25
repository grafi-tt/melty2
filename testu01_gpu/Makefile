# testu01_gpu: make builds build/libtestu01_gpu.a and the example build/splitmix_bigcrush.
#   HIPCC     the HIP compiler (default hipcc)
#   HIP_ARCH  the GPU architectures of the kernels of rocThrust (the kernels of the tests are compiled at
#             run time for the device), e.g. gfx1030 or "gfx1030 --offload-arch=gfx1100"
#   TESTU01   the installation prefix of TestU01 (its headers and the probdist and mylib libraries)
# A program using the library links build/libtestu01_gpu.a with -lprobdist -lmylib -lhiprtc -lrocfft (see the
# rule of the example).
HIPCC ?= hipcc
HIP_ARCH ?= gfx1030
TESTU01 ?= /usr/local
B := build

LIBS := -L$(TESTU01)/lib -Wl,-rpath,$(TESTU01)/lib -lprobdist -lmylib -lhiprtc -lrocfft

.PHONY: all clean
all: $(B)/libtestu01_gpu.a $(B)/splitmix_bigcrush

# the device code and the parameters header are embedded into the library as raw string literals
$(B)/device.inc: src/device.hpp | $(B)
	{ printf 'R"TESTU01_GPU('; cat $<; printf ')TESTU01_GPU"\n'; } > $@
$(B)/params.inc: include/testu01_gpu_params.hpp | $(B)
	{ printf 'R"TESTU01_GPU('; cat $<; printf ')TESTU01_GPU"\n'; } > $@

$(B)/testu01_gpu.o: src/testu01_gpu.hip src/device.hpp include/testu01_gpu.hpp include/testu01_gpu_params.hpp $(B)/device.inc $(B)/params.inc | $(B)
	$(HIPCC) -std=c++17 --offload-arch=$(HIP_ARCH) -O3 -Wall -Iinclude -Isrc -I$(B) -I$(TESTU01)/include -c $< -o $@
$(B)/libtestu01_gpu.a: $(B)/testu01_gpu.o
	rm -f $@
	ar rcs $@ $^

$(B)/splitmix_bigcrush.o: examples/splitmix_bigcrush.cpp include/testu01_gpu.hpp include/testu01_gpu_params.hpp | $(B)
	$(HIPCC) -std=c++17 --offload-arch=$(HIP_ARCH) -O2 -Wall -Iinclude -c $< -o $@
$(B)/splitmix_bigcrush: $(B)/splitmix_bigcrush.o $(B)/libtestu01_gpu.a
	$(HIPCC) --offload-arch=$(HIP_ARCH) $^ $(LIBS) -o $@

$(B):
	mkdir -p $@
clean:
	rm -rf $(B)
