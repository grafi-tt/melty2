CC = cc
LD = cc
CFLAGS = -O3 -march=native -fno-stack-protector -Wall -Wextra -Wpedantic -Werror
LDFLAGS =

.PHONY: all
all: melty2.so test/bigcrush test/bigcrush_bitrev

melty2.so: melty2.o
	$(LD) -shared $(LDFLAGS) $^ -o $@

test/bigcrush: test/bigcrush.o melty2.o
	$(LD) $(LDFLAGS) -ltestu01 -lprobdist -lmylib $^ -o $@

test/bigcrush_bitrev: test/bigcrush_bitrev.o melty2.o
	$(LD) $(LDFLAGS) -ltestu01 -lprobdist -lmylib $^ -o $@

melty2.o: melty2.c include/melty2.h
	$(CC) $(CFLAGS) -fPIC -std=c99 -I./include $< -c -o $@

test/bigcrush.o: test/bigcrush.c include/melty2.h
	$(CC) $(CFLAGS) -std=c99 -I./include $< -c -o $@

test/bigcrush_bitrev.o: test/bigcrush.c include/melty2.h
	$(CC) $(CFLAGS) -std=c99 -I./include -DBITREV $< -c -o $@

.PHONY: clean
clean:
	rm -rf \
		melty2.so \
		test/bigcrush \
		test/bigcrush_bitrev \
		melty2.o \
		test/bigcrush.o \
		test/bigcrush_bitrev.o
