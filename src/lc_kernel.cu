#include "lc_kernel.hu"

__device__ const double probs[7] = {0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833};

__device__ uint64_t dev_extract_bits(const uint8_t*__restrict data, int start, int bits){
    uint64_t value = 0;
    int byte = start / 8;
    int bit = start % 8;
    int bits_left = bits;
    while (bits_left > 0){
        int bits_to_extract = (8 - bit < bits_left) ? 8 - bit : bits_left;
        uint64_t mask = (1 << bits_to_extract) - 1;
        value = (value << bits_to_extract) | ((data[byte] >> (8 - bit - bits_to_extract)) & mask);
        bits_left -= bits_to_extract;
        bit = 0;
        byte++;
    }
    return value;
}

__device__ uint64_t dev_uintxor(uint64_t num) {
    num ^= (num >> 32);
    num ^= (num >> 16);
    num ^= (num >> 8);
    num ^= (num >> 4);
    num ^= (num >> 2);
    num ^= (num >> 1);
    return num & 1;
}

__device__ uint64_t dev_trailing_zeros(uint64_t num) {
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

__device__ uint64_t dev_reverse_uint64_t(uint64_t num, uint64_t len) {
    uint64_t rev_n = 0;
    uint64_t i;
    for (i = 0; i < 64; i++) {
        uint64_t bit = (num & (1 << i)) >> i;
        rev_n |= bit << (64 - 1 - i);
    }
    rev_n >>= (64 - len);
    return rev_n;
} 

__device__ uint64_t dev_complexity(uint64_t sequence, uint64_t length) {
    if(sequence == 0){
        return 1;
    }
    uint64_t N = length;

    uint64_t k = dev_trailing_zeros(sequence);
    uint64_t F = (1 << (k + 1)) | 1;
    uint64_t G = 1;
    uint64_t l = k + 1;
    uint64_t a = k;
    uint64_t b = 0;
    uint64_t n = k;

    for(n = k + 1; n < N; n++){
        uint64_t d = dev_uintxor(sequence & (F << (n - l)));
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
                b = n - l +1;
            }
        }
    }

    return l;
}

__device__ double dev_lc_test(uint8_t* data, uint64_t data_size, uint64_t bit_sequence_len) {
    uint64_t bins[7] = {0, 0, 0, 0, 0, 0, 0};
    uint64_t sequences_num = (data_size * 8) / bit_sequence_len;
    uint64_t i;

    double s_one = (bit_sequence_len & 1) ? -1.0 : 1.0;
    double mi = (double) (bit_sequence_len / 2.0); /// ?
    mi += (9.0 - s_one) / 36.0;
    mi -= ((bit_sequence_len / 3.0) + (2.0 / 9.0)) / pow(2.0, bit_sequence_len);

    for (i = 0; i < sequences_num; i++) {
        uint64_t starting_bit = i * bit_sequence_len;
        uint64_t sequence = dev_extract_bits(data, starting_bit, bit_sequence_len);
        sequence = dev_reverse_uint64_t(sequence, bit_sequence_len); /// [KERNEIZED]
        uint64_t lc = dev_complexity(sequence, bit_sequence_len);

        double ti = s_one * ((double) lc - mi) + 2.0 / 9.0;
        bins[0] += (ti <= -2.5) ? 1u : 0;
        bins[1] += (ti > -2.5 && ti <= -1.5) ? 1u : 0;
        bins[2] += (ti > -1.5 && ti <= -0.5) ? 1u : 0;
        bins[3] += (ti > -0.5 && ti <= 0.5) ? 1u : 0;
        bins[4] += (ti > 0.5 && ti <= 1.5) ? 1u : 0;
        bins[5] += (ti > 1.5 && ti <= 2.5) ? 1u : 0;
        bins[6] += (ti > 2.5) ? 1u : 0;
    }

    double chi = 0.0;
    for (i = 0; i < 7; i++) {
        double expected = probs[i] * sequences_num;
        double enumerator = pow((double) bins[i] - expected, 2.0);
        chi += enumerator / expected;
    }
    return chi;
}

__global__ void vectorSum(const uint8_t*__restrict data_in, double*__restrict data_out, int bit_sequence_len, int data_n, int data_size){
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Boundary check
  if (tid < data_n){
    int offset = tid * data_size;
    uint8_t* data = (uint8_t*)&data_in[offset];
    double chi = dev_lc_test(data, data_size, bit_sequence_len);
    data_out[tid] = chi;
  }
}