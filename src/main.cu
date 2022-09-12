#include "lc_kernel.hu"
#include "linear_complexity.hpp"
#include "io.hpp"
#include <iostream>
#include <vector>
#include <assert.h>

int main() {
  const int data_n = 1024 * 64;
  const int data_size = 1024;
  const int threads_per_block = 256;
  const int blocks_per_grid = (data_n + threads_per_block - 1) / threads_per_block;

  std::vector<uint8_t> h_data_in(data_n * data_size);
  std::vector<uint64_t> h_data_out(data_n);

  for(auto &i : h_data_in) {
    i = rand() % 256;
  }

  print_bit_data(h_data_in, 31, 8);


  uint8_t *d_data_in;
  uint64_t *d_data_out;
  cudaMalloc(&d_data_in, data_n * data_size);
  cudaMalloc(&d_data_out, data_n);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_data_in, h_data_in.data(), data_n * data_size, cudaMemcpyHostToDevice);

  vectorSum<<<blocks_per_grid, threads_per_block>>>(d_data_in, d_data_out, data_n, data_size);

  cudaMemcpy(h_data_out.data(), d_data_out, data_n, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 10; i++) {
    print_uint64(h_data_out[i], 31);
    uint64_t bits = extract_bits(h_data_in, i * data_size, 31);
    print_uint64(bits, 31);
    printf("\n");
  }

  // Free memory on device
  cudaFree(d_data_in);
  cudaFree(d_data_out);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}