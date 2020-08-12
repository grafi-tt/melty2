#!/usr/bin/env python3
import hashlib
import sys

import msgpack


class TestGenerator():
    def __init__(self, f, name):
        self._f = f
        self._hctx = hashlib.blake2b()
        self._packer = msgpack.Packer()

        self._f.write(f"static void {name}(mely2_key_t *key)" " {\n")
        self._f.write("    melty2_seeder_t seeder;\n")
        self._f.write("    melty2_init_seeder(&seeder);\n")

    def seed_null(self):
        self._hctx.update(self._packer.pack(None))
        self._f.write("    melty2_seed_null(&seeder);\n")

    def seed_bool(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_bool(&seeder, {1 if s else 0});\n')

    def seed_uint(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_uint(&seeder, {s});\n')

    def seed_int(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_int(&seeder, {s});\n')

    def seed_float(self, s):
        self._packer.use_single_float = True
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_float(&seeder, {s});\n')

    def seed_double(self, s):
        self._packer.use_single_float = False
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_double(&seeder, {s});\n')

    def seed_str(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_str(&seeder, "{s}", {len(s)});\n')

    def seed_bin(self, s):
        s = s.decode()
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_str(&seeder, "{s}", {len(s)});\n')

    def generate(self):
        self._f.write('}\n')
        return self._hctx.hexdigest()[:384]


def gen_test_case(f, name, *seeds, use_fp32=False):
    gen = TestGenerator(f, name)
    for s in seeds:
        if s is None:
            gen.seed_null()
        elif isinstance(s, bool):
            gen.seed_bool(s)
        elif isinstance(s, int):
            if s >= 0:
                gen.seed_uint(s)
            else:
                gen.seed_int(s)
        elif isinstance(s, float):
            if use_fp32:
                gen.seed_float(s)
            else:
                gen.seed_double(s)
        elif isinstance(s, str):
            gen.seed_str(s)
        elif isinstance(s, bytes):
            gen.seed_bin(s)
        else:
            raise TypeError(f"Invlid seed type: {type(s)}")
    expected_hex = gen.generate()
    f.write('\n')
    return name, expected_hex


def gen_tests(f):
    cases = [
        gen_test_case(f, 'test_empty'),
        gen_test_case(f, 'test_null', None),
        gen_test_case(f, 'test_0', 0),
        gen_test_case(f, 'test_1', 1),
        gen_test_case(f, 'test_127', 127),
        gen_test_case(f, 'test_128', 128),
        gen_test_case(f, 'test_255', 255),
        gen_test_case(f, 'test_256', 256),
        gen_test_case(f, 'test_12345', 12345),
        gen_test_case(f, 'test_65535', 65535),
        gen_test_case(f, 'test_65536', 65536),
        gen_test_case(f, 'test_pi_10d', 3141592653),
        gen_test_case(f, 'test_2to32m1', 2 ** 32 - 1),
        gen_test_case(f, 'test_2to32', 2 ** 32),
        gen_test_case(f, 'test_sqrt2_10d_pi_10d', 14142135623141592653),
        gen_test_case(f, 'test_2to64m1', 2 ** 64 - 1),
    ]

    f.write('typedef struct {\n')
    f.write('    const char* name;\n')
    f.write('    void(*)(mely2_key_t *key) fn;\n')
    f.write('    const char* expected_hex;\n')
    f.write('} TestCase;\n')
    f.write('\n')
    f.write('static const TestCase test_cases[] = {\n')
    for name, expected_hex in cases:
        f.write('    {' f'"{name}" {name}, "{expected_hex}"' '},\n')
    f.write('};\n')


if __name__ == '__main__':
    gen_tests(sys.stdout)
