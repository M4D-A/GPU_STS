#include "linear_complexity.hpp"
#include "io.hpp"
#include <math.h>

uint8_t byterev[256] = { 
  0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0, 
  0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 
  0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4, 
  0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc, 
  0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 
  0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa, 
  0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6, 
  0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 
  0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1, 
  0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9, 
  0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 
  0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd, 
  0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3, 
  0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 
  0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7, 
  0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff, 
};

double probs[7] = {0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833};

uint64_t extract_reverse_bits(const uint8_t* data, uint64_t start, uint64_t bits){
    uint64_t end = start + bits;

    uint64_t s_byte = start / 8;
    uint64_t e_byte = end / 8;

    uint64_t s_bit = start % 8;
    uint64_t e_bit = end % 8;

    uint8_t right_bit_mask = (1 << (e_bit)) -1;

    uint64_t output = 0;
    uint64_t i;
    for (i = e_byte; i >= s_byte; i--){
        uint8_t current_byte = byterev[data[i]];

        current_byte &= (i == e_byte) ? right_bit_mask : 0xff;

        current_byte >>= (i == s_byte) ? (s_bit) : 0;
        output <<= (i == s_byte) ? (8-s_bit) : 8;

        output |= current_byte;
        if(i == s_byte) break;
    }
    return output;
}

uint64_t uintxor(uint64_t num) {
    return __builtin_parity(num);
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

uint64_t complexity(uint64_t sequence, uint64_t length) {
    if(sequence == 0){
        return 1;
    }
    uint64_t N = length;

    uint64_t k = trailing_zeros(sequence);
    uint64_t F = (1 << (k + 1)) | 1;
    uint64_t G = 1;
    uint64_t l = k + 1;
    uint64_t a = k;
    uint64_t b = 0;
    uint64_t n = k;

    for(n = k + 1; n < N; n++){
        uint64_t d = uintxor(sequence & (F << (n - l)));
        if(d == 0){
            b+=1;
        }
        else{
            if(2*l > n){
                F ^= (G << (a - b));
                b += 1;
            }
            else{
                uint64_t T = F;
                F = (F << (b - a)) ^ G;
                l = n + 1 - l;
                G = T;
                a = b;
                b = n - l + 1;
            }
        }
    }

    return l;
}

double lc_test(const std::vector<uint8_t> &data, uint64_t bit_sequence_len) {

    uint64_t bins[7] = {0, 0, 0, 0, 0, 0, 0};
    uint64_t sequences_num = (data.size()* 8) / bit_sequence_len;
    uint64_t i;

    double s_one = (bit_sequence_len & 1) ? -1.0 : 1.0;
    double mi = (double) (bit_sequence_len / 2.0); /// ?
    mi += (9.0 - s_one) / 36.0;
    mi -= ((bit_sequence_len / 3.0) + (2.0 / 9.0)) / pow(2.0, bit_sequence_len);

    for (i = 0; i < sequences_num; i++) {
        uint64_t starting_bit = i * bit_sequence_len;
        uint64_t sequence = extract_reverse_bits(data.data(), starting_bit, bit_sequence_len);
        uint64_t lc = complexity(sequence, bit_sequence_len);

        double ti = s_one * ((double) lc - mi) + 2.0 / 9.0;
        bins[0] += (ti <= -2.5) ? 1u : 0;
        bins[1] += (ti > -2.5 && ti <= -1.5) ? 1u : 0;
        bins[2] += (ti > -1.5 && ti <= -0.5) ? 1u : 0;
        bins[3] += (ti > -0.5 && ti <= 0.5) ? 1u : 0;
        bins[4] += (ti >  0.5 && ti <= 1.5) ? 1u : 0;
        bins[5] += (ti >  1.5 && ti <= 2.5) ? 1u : 0;
        bins[6] += (ti >  2.5) ? 1u : 0;
    }

    double chi = 0.0;
    for (i = 0; i < 7; i++) {
        double expected = probs[i] * sequences_num;
        double enumerator = powl((double) bins[i] - expected, 2.0);
        chi += enumerator / expected;
    }
    return chi;
}

