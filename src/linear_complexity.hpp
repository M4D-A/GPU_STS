#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>


uint64_t extract_reverse_bits(const uint8_t* data, uint64_t start, uint64_t bits);
uint64_t uintxor(uint64_t num);
uint64_t trailing_zeros(uint64_t num);
uint64_t reverse_uint64_t(uint64_t num, uint64_t len);
uint64_t complexity(uint64_t sequence, uint64_t length);
double lc_test(const std::vector<uint8_t> &data, uint64_t bit_sequence_len);