# SPDX-License-Identifier: MIT
# make cpu : build/{prof,sa,weakkeys}_cpu with the CPU metric, and build/check
# make gpu : build/{prof,sa,weakkeys}_gpu with the HIP metric (HIPCC, HIP_ARCH)
CC ?= cc
CFLAGS ?= -O3 -march=native
HIPCC ?= hipcc
HIP_ARCH ?= gfx1030
B := build
ROOT := ..
INC := -I. -I$(ROOT)/include
TOOLS := prof sa weakkeys

.PHONY: cpu gpu clean
cpu: $(TOOLS:%=$(B)/%_cpu) $(B)/check
gpu: $(TOOLS:%=$(B)/%_gpu)

$(B)/%.o: %.c arx.h metric.h keymod.h | $(B)
	$(CC) -std=c99 $(CFLAGS) $(INC) -c $< -o $@
$(B)/melty2_init.o: $(ROOT)/src/melty2_init.c | $(B)
	$(CC) -std=c99 $(CFLAGS) $(INC) -c $< -o $@
$(B)/metric_gpu.o: metric_gpu.hip arx.h metric.h | $(B)
	$(HIPCC) --offload-arch=$(HIP_ARCH) -O3 $(INC) -c $< -o $@

COMMON := $(B)/metric_common.o $(B)/melty2_init.o
$(B)/%_cpu: $(B)/%.o $(B)/metric_cpu.o $(COMMON)
	$(CC) $^ -o $@ -lm
$(B)/%_gpu: $(B)/%.o $(B)/metric_gpu.o $(COMMON)
	$(HIPCC) --offload-arch=$(HIP_ARCH) $^ -o $@ -lm

$(B)/check: check.c arx.h | $(B)
	$(CC) -std=c99 $(CFLAGS) $(INC) $< -o $@

$(B):
	mkdir -p $@
clean:
	rm -rf $(B)
