#include "io.hpp"

std::vector<uint8_t> read_file(std::string file_name){
    std::ifstream file(file_name, std::ios::binary);
    std::vector<uint8_t> data;
    if (file.is_open()){
        file.seekg(0, std::ios::end);
        data.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read((char*)data.data(), data.size());
        file.close();
    }
    return data;
}

std::vector<std::string> files_in_dir(std::string dir_name, std::string ext){
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(dir_name)){
        if (entry.path().extension() == ext){
            files.push_back(entry.path().string());
        }
    }
    return files;
}




uint64_t extract_bits(std::vector<uint8_t> data, int start, int bits){
    uint64_t value = 0;
    int byte = start / 8;
    int bit = start % 8;
    int bits_left = bits;
    while (bits_left > 0){
        int bits_to_extract = std::min(8 - bit, bits_left);
        uint64_t mask = (1 << bits_to_extract) - 1;
        value = (value << bits_to_extract) | ((data[byte] >> (8 - bit - bits_to_extract)) & mask);
        bits_left -= bits_to_extract;
        bit = 0;
        byte++;
    }
    return value;
}

uint64_t uintxor(uint64_t num) {
    num ^= (num >> 32);
    num ^= (num >> 16);
    num ^= (num >> 8);
    num ^= (num >> 4);
    num ^= (num >> 2);
    num ^= (num >> 1);
    return num & 1;
}

uint64_t trailing_zeros(uint64_t num) {
    num = (num & (~(num - 1)));
    uint64_t c = 64;
    c = (num & 0x00000000FFFFFFFF) ? c - 32 : c;
    c = (num & 0x0000FFFF0000FFFF) ? c - 16 : c;
    c = (num & 0x00FF00FF00FF00FF) ? c - 8 : c;
    c = (num & 0x0F0F0F0F0F0F0F0F) ? c - 4 : c;
    c = (num & 0x3333333333333333) ? c - 2 : c;
    c = (num & 0x5555555555555555) ? c - 1 : c;
    c = (num                     ) ? c - 1 : c;
    return c;
}

uint64_t reverse_uint64_t(uint64_t num, uint64_t len) {
    uint64_t rev_n = 0;
    uint64_t i;
    for (i = 0; i < 64; i++) {
        uint64_t bit = (num & (1 << i)) >> i;
        rev_n |= bit << (64 - 1 - i);
    }
    rev_n >>= (64 - len);
    return rev_n;
} 

void print_hex_data(std::vector<uint8_t> data){
    for (auto byte : data){
        printf("%02x ", byte);
    }
    printf("\n");
}

void print_bit_data(std::vector<uint8_t> data, int bits){
    const int break_point = 64;
    int break_at = 0;
    for (int i = 0; i < data.size() * 8; i++){
        printf("%d", (data[i / 8] >> (7- (i%8)) & 1));
        if ((i + 1) % bits == 0){
            break_at += bits;
            if(break_at >= break_point){
                printf("\n");
                break_at = 0;
            }
            else{
                printf(" ");
            }
        }
    }
    if (break_at != 0 || break_point >= 64){
        printf("\n");
    }
}

void print_uint64(uint64_t value, int bits){
    for (int i = 0; i < bits; i++){
        printf("%ld", (value >> (bits - i - 1)) & 1);
    }
    printf("\n");
}