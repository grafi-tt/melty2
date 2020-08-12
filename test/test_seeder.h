/* This file is generated by gen_seeder_tests.py */

static void test_empty(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_initkey(seeder, key);
}

static void test_null(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_null(seeder);
    melty2_initkey(seeder, key);
}

static void test_0(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(0));
    melty2_initkey(seeder, key);
}

static void test_1(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(1));
    melty2_initkey(seeder, key);
}

static void test_127(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(127));
    melty2_initkey(seeder, key);
}

static void test_128(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(128));
    melty2_initkey(seeder, key);
}

static void test_255(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(255));
    melty2_initkey(seeder, key);
}

static void test_256(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(256));
    melty2_initkey(seeder, key);
}

static void test_12345(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(12345));
    melty2_initkey(seeder, key);
}

static void test_65535(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(65535));
    melty2_initkey(seeder, key);
}

static void test_65536(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(65536));
    melty2_initkey(seeder, key);
}

static void test_pi10d(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(3141592653));
    melty2_initkey(seeder, key);
}

static void test_2to32m1(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(4294967295));
    melty2_initkey(seeder, key);
}

static void test_2to32(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(4294967296));
    melty2_initkey(seeder, key);
}

static void test_pi19d(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(3141592653589693238));
    melty2_initkey(seeder, key);
}

static void test_2to64m1(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_uint(seeder, UINT64_C(18446744073709551615));
    melty2_initkey(seeder, key);
}

static void test_m1(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-1));
    melty2_initkey(seeder, key);
}

static void test_m32(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-32));
    melty2_initkey(seeder, key);
}

static void test_m33(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-33));
    melty2_initkey(seeder, key);
}

static void test_m256(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-256));
    melty2_initkey(seeder, key);
}

static void test_m257(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-257));
    melty2_initkey(seeder, key);
}

static void test_m12345(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-12345));
    melty2_initkey(seeder, key);
}

static void test_m65536(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-65536));
    melty2_initkey(seeder, key);
}

static void test_m65537(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-65537));
    melty2_initkey(seeder, key);
}

static void test_m_pi10d(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-3141592653));
    melty2_initkey(seeder, key);
}

static void test_m_2to32(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-4294967296));
    melty2_initkey(seeder, key);
}

static void test_m_2to32p1(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-4294967297));
    melty2_initkey(seeder, key);
}

static void test_m_pi19d(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_int(seeder, INT64_C(-3141592653589693238));
    melty2_initkey(seeder, key);
}

static void test_float(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_float(seeder, 1.23);
    melty2_initkey(seeder, key);
}

static void test_double(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_double(seeder, 1.23);
    melty2_initkey(seeder, key);
}

static void test_str0(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "", 0);
    melty2_initkey(seeder, key);
}

static void test_fixstr(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "spam", 4);
    melty2_initkey(seeder, key);
}

static void test_str31(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "spamspamspamspamspamspamspamham", 31);
    melty2_initkey(seeder, key);
}

static void test_str32(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "spamspameggeggeggeggeggeggeggegg", 32);
    melty2_initkey(seeder, key);
}

static void test_str255(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 255);
    melty2_initkey(seeder, key);
}

static void test_str256(melty2_key *key) {
    melty2_seeder seeder[1];
    melty2_initseeder(seeder);
    melty2_seed_str(seeder, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 256);
    melty2_initkey(seeder, key);
}

typedef struct {
    const char* name;
    void (*fn)(melty2_key *key);
    const char* expected_hex;
} TestCase;

static const TestCase test_cases[] = {
    {"test_empty", test_empty,
     "786a02f742015903c6c6fd852552d272912f4740e1584761"},
    {"test_null", test_null,
     "63b306835317e1a8c24e4d481289a0d48aa5699b0d6d5ccd"},
    {"test_0", test_0,
     "2fa3f686df876995167e7c2e5d74c4c7b6e48f8068fe0e44"},
    {"test_1", test_1,
     "9545ba37b230d8a2e716c4707586542780815b7c4088edcb"},
    {"test_127", test_127,
     "e92b41dd767e71d505805bcb09c6fdb3f0954b37b06e9cd7"},
    {"test_128", test_128,
     "5aeff8431522935d5e31f44be0d5f2854ba158a83b67065c"},
    {"test_255", test_255,
     "64528de49dcdd7cebe1b7f2b18372332255d67c598b45c0f"},
    {"test_256", test_256,
     "e4451e07b45a3431d251c689320258606092e3fb861954a2"},
    {"test_12345", test_12345,
     "08ebd3d2c85d2b9af6eb33b75d16266583336dfae779cde3"},
    {"test_65535", test_65535,
     "8bbca3bc2cb94a4ea061e42bc0060b0849ff7426bd7c3168"},
    {"test_65536", test_65536,
     "9f98b0f8307963aa34f3272102e5b69b72c23510c36803e8"},
    {"test_pi10d", test_pi10d,
     "1b9e63e491215e9bb96502b8d55728e31d7686b049c7b3a2"},
    {"test_2to32m1", test_2to32m1,
     "9fd5bfca3617be2aeaf36aac1aaec1d19016015ce4ee7fea"},
    {"test_2to32", test_2to32,
     "bcd981776095b3757a106f95936514c0dd481824752b591a"},
    {"test_pi19d", test_pi19d,
     "1f439de74a433a815c6541622d919f4c232aafb4d29831fa"},
    {"test_2to64m1", test_2to64m1,
     "011dd784631c27d6d93316319cf1e40f242c6b3b62a856fa"},
    {"test_m1", test_m1,
     "eb65152dcb7b3371d6399005e2e0fac3e0858c5c51448384"},
    {"test_m32", test_m32,
     "67d0ae5403416faf5c21190b82df6677b8d30b3fc3803347"},
    {"test_m33", test_m33,
     "b718d304e89b905ba95a083673b0f79765c20a6a3c504e0b"},
    {"test_m256", test_m256,
     "067762d6b0dc69d98f4e83339c51d11fbdc8f5728ddc06b4"},
    {"test_m257", test_m257,
     "95ca2f2e2887a06e627eef889598eda2fe255dc12e935241"},
    {"test_m12345", test_m12345,
     "6685e154af0501fd7f6c66d3716d1e1e7ef3fcb4cd0343f2"},
    {"test_m65536", test_m65536,
     "9ebdf66ed81192f0eb3a257cc635425ddfa6563b52f442cf"},
    {"test_m65537", test_m65537,
     "82dae309eed889bd829aa9c2f3d8b6f638b4e49d6c6b07ca"},
    {"test_m_pi10d", test_m_pi10d,
     "4b0b3173c200799afa7a601049ada9fec16ee85d8994837f"},
    {"test_m_2to32", test_m_2to32,
     "c35e4c89fad17d56f0a1acab27155ad10ac6f000cc12cfc9"},
    {"test_m_2to32p1", test_m_2to32p1,
     "b5ee5300e6cb130aa4b643a3944d98cb2b534d154d7db178"},
    {"test_m_pi19d", test_m_pi19d,
     "92b8c2b12aa43faadde45ee671efc2a11cd79fe637756ff8"},
    {"test_float", test_float,
     "901f50a481165cba5334c86aef50ad78ebf66eac909cfc96"},
    {"test_double", test_double,
     "70e9631695b510e31c349ad51085db5c7b7234e48124c9b3"},
    {"test_str0", test_str0,
     "d17f0a6aa8cc2f282a33fac0c32fc710368946831fb5bee8"},
    {"test_fixstr", test_fixstr,
     "a8fe3171ca3159b0952c019d61765f67bd2184f251c73e6f"},
    {"test_str31", test_str31,
     "dd7f02614cdacdcf0d0fb93ae308c6db50c2df4c6406e8d3"},
    {"test_str32", test_str32,
     "2aea929329398767bdd0942394f880e1e036fdc0fb9538fa"},
    {"test_str255", test_str255,
     "37a907eb4fd63b3669035610bf06db1c7cc4e969226f9619"},
    {"test_str256", test_str256,
     "0be7ea1e2688a410670b1b5dee57808e86696f7ffc182c77"},
};
