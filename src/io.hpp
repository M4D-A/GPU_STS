#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

std::vector<uint8_t> read_file(std::string file_name);

std::vector<std::string> files_in_dir(std::string dir_name, std::string ext);

void print_hex_data(std::vector<uint8_t> data);

void print_bit_data(std::vector<uint8_t> data, int bits, int words);

void print_uint64(uint64_t value, int bits);