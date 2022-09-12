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

void print_hex_data(std::vector<uint8_t> data){
    for (auto byte : data){
        printf("%02x ", byte);
    }
    printf("\n");
}

void print_bit_data(std::vector<uint8_t> data, int bits, int words){
    const int break_point = 64;
    int word_count = 0;
    int break_at = 0;
    for (int i = 0; i < data.size() * 8; i++){
        printf("%d", (data[i / 8] >> (7- (i%8)) & 1));
        if ((i + 1) % bits == 0){
            word_count++;
            if (word_count > words){
                printf("\n");
                return;
            }
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