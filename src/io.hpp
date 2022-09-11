#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

std::vector<uint8_t> read_file(std::string file_name);

std::vector<std::string> files_in_dir(std::string dir_name, std::string ext);




uint64_t extract_bits(std::vector<uint8_t> data, int start, int bits);

uint64_t uintxor(uint64_t num);

uint64_t trailing_zeros(uint64_t num);

uint64_t reverse_uint64_t(uint64_t num, uint64_t len);

void print_hex_data(std::vector<uint8_t> data);

void print_bit_data(std::vector<uint8_t> data, int bits);

void print_uint64(uint64_t value, int bits);