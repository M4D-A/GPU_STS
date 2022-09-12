#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

uint64_t extract_bits(std::vector<uint8_t> data, int start, int bits);

uint64_t uintxor(uint64_t num);

uint64_t trailing_zeros(uint64_t num);

uint64_t reverse_uint64_t(uint64_t num, uint64_t len);

uint64_t complexity(uint64_t sequence, uint64_t length);

long double lc_test(std::vector<uint8_t> data, uint64_t bit_sequence_len);