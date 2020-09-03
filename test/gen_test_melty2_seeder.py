#!/usr/bin/env python3
# This script generates test_melty2_seeder.h
import hashlib

import msgpack


class TestGenerator():
    def __init__(self, f, name, use_fp32):
        self._f = f
        self._hctx = hashlib.blake2b()
        self._packer = msgpack.Packer(use_single_float=use_fp32)

        self._f.write(f"static void {name}(melty2_key *key)" " {\n")
        self._f.write("    melty2_seeder seeder[1];\n")
        self._f.write("    melty2_initseeder(seeder);\n")

    def seed_null(self):
        self._hctx.update(self._packer.pack(None))
        self._f.write("    melty2_seed_null(seeder);\n")

    def seed_bool(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_bool(seeder, {1 if s else 0});\n')

    def seed_uint(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_uint(seeder, UINT64_C({s}));\n')

    def seed_int(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_int(seeder, INT64_C({s}));\n')

    def seed_float(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_float(seeder, {s}f);\n')

    def seed_double(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_double(seeder, {s});\n')

    def seed_str(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_strwithlen(seeder, "{s}", {len(s)});\n')

    def seed_bin(self, s):
        self._hctx.update(self._packer.pack(s))
        self._f.write(f'    melty2_seed_bin(seeder, "{s.decode()}", {len(s)});\n')

    def generate(self):
        self._f.write('    melty2_initkey(key, seeder);\n')
        self._f.write('}\n')
        return self._hctx.hexdigest()[:48]


def gen_test_case(f, name, *seeds, use_fp32=False):
    gen = TestGenerator(f, name, use_fp32)
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
        gen_test_case(f, 'test_pi10d', 3141592653),
        gen_test_case(f, 'test_2to32m1', 2 ** 32 - 1),
        gen_test_case(f, 'test_2to32', 2 ** 32),
        gen_test_case(f, 'test_pi19d', 3141592653589693238),
        gen_test_case(f, 'test_2to64m1', 2 ** 64 - 1),
        gen_test_case(f, 'test_m1', -1),
        gen_test_case(f, 'test_m32', -32),
        gen_test_case(f, 'test_m33', -33),
        gen_test_case(f, 'test_m256', -256),
        gen_test_case(f, 'test_m257', -257),
        gen_test_case(f, 'test_m12345', -12345),
        gen_test_case(f, 'test_m65536', -65536),
        gen_test_case(f, 'test_m65537', -65537),
        gen_test_case(f, 'test_m_pi10d', -3141592653),
        gen_test_case(f, 'test_m_2to32', -(2 ** 32)),
        gen_test_case(f, 'test_m_2to32p1', -(2 ** 32 + 1)),
        gen_test_case(f, 'test_m_pi19d', -3141592653589693238),
        # INT64_C doesn't work well for INT64_MIN
        # gen_test_case(f, 'test_m_2to63', -(2 ** 63)),
        gen_test_case(f, 'test_float', 1.23, use_fp32=True),
        gen_test_case(f, 'test_double', 1.23, use_fp32=False),
        gen_test_case(f, 'test_str0', ''),
        gen_test_case(f, 'test_fixstr', 'spam'),
        gen_test_case(f, 'test_str31', 'spam' * 7 + 'ham'),
        gen_test_case(f, 'test_str32', 'spam' * 2 + 'egg' * 8),
        gen_test_case(f, 'test_str255', 'x' * 255),
        gen_test_case(f, 'test_str256', 'x' * 256),
        # Avoid too large literal
        # gen_test_case(f, 'test_str65535', 'x' * 65535),
        # gen_test_case(f, 'test_str65536', 'x' * 65536),
        gen_test_case(f, 'test_bin0', b''),
        gen_test_case(f, 'test_bin255', b'x' * 255),
        gen_test_case(f, 'test_bin256', b'x' * 256),
        # Avoid too large literal
        # gen_test_case(f, 'test_bin65535', b'x' * 65535),
        # gen_test_case(f, 'test_bin65536', b'x' * 65536),
    ]

    f.write('typedef struct {\n')
    f.write('    const char* name;\n')
    f.write('    void (*fn)(melty2_key *key);\n')
    f.write('    const char* expected_hex;\n')
    f.write('} TestCase;\n')
    f.write('\n')
    f.write('static const TestCase test_cases[] = {\n')
    for name, expected_hex in cases:
        f.write('    {' f'"{name}", {name},\n     "{expected_hex}"' '},\n')
    f.write('};\n')


if __name__ == '__main__':
    with open('test_melty2_seeder.h', 'w') as f:
        f.write('/* This file is generated by gen_test_melty2_seeder.py */\n')
        f.write('\n')
        gen_tests(f)
