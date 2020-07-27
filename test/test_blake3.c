#include "melty2.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t input_len;
    const char* hash_hex;
} TestVector;

static const TestVector vectors[] = {
    {0,      "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
    {1,      "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
    {1023,   "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"},
    {1024,   "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
    {1025,   "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"},
    {2048,   "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"},
    {2049,   "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"},
    {3072,   "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"},
    {3073,   "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"},
    {4096,   "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"},
    {4097,   "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995"},
    {5120,   "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833"},
    {5121,   "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff"},
    {6144,   "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205"},
    {6145,   "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"},
    {7168,   "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a"},
    {7169,   "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817"},
    {8192,   "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"},
    {8193,   "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b"},
    {16384,  "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4"},
    {31744,  "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47"},
    {102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"},
};

int test(const TestVector* vector) {
    char expected[32];
    for (int i = 0; i < 32; i++) {
        unsigned int c;
        if (sscanf(&vector->hash_hex[i * 2], "%02x", &c) != 1) {
            fputs("sscanf failed\n", stderr);
            return -1;
        }
        expected[i] = (char)c;
    }

    char* input;
    if (!(input = malloc(vector->input_len))) {
        perror("malloc failed: ");
        return -1;
    }
    for (uint64_t i = 0; i < vector->input_len; ++i) {
        input[i] = (char)(i % 251);
    }

    melty2_name name;
    melty2_initname_withlen(&name, input, vector->input_len);
    free(input);

    char result[32];
    for (size_t i = 0; i < sizeof(name.v_) / sizeof(uint32_t); ++i) {
        result[i * sizeof(uint32_t)] = (char)(unsigned char)(name.v_[i]);
        result[i * sizeof(uint32_t) + 1] = (char)(unsigned char)(name.v_[i] >> 8);
        result[i * sizeof(uint32_t) + 2] = (char)(unsigned char)(name.v_[i] >> 16);
        result[i * sizeof(uint32_t) + 3] = (char)(unsigned char)(name.v_[i] >> 24);
    }

    int err = !!memcmp(result, expected, sizeof(expected));
    if (err) {
        fprintf(stderr, "test failed: input_len=%"PRIu64"\n", vector->input_len);
    }
    return err;
}

int main() {
    int n_err = 0;
    for (size_t t = 0; t < sizeof(vectors) / sizeof(TestVector); ++t) {
        int err = test(&vectors[t]);
        if (err < 0) return 63;
        n_err += err;
    }
    return n_err;
}
