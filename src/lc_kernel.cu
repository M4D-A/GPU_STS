#include "lc_kernel.hu"

__device__ uint64_t extract_bits_dev(const uint8_t*__restrict data, int start, int bits){
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

__global__ void vectorSum(const uint8_t*__restrict data_in, uint64_t*__restrict data_out, int data_n, int data_size){
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < data_n){
    uint64_t bits = extract_bits_dev(data_in, tid * data_size, 31);
    data_out[tid] = bits;
  }
}