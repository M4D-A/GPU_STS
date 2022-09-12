#include "lc_kernel.hu"
#include "linear_complexity.hpp"
#include "io.hpp"
#include <iostream>
#include <vector>
#include <assert.h>

int main() {
  const int bytes = 1024 * 1024 * 1024;
  const int data_n = 2560;
  const int data_size = bytes / data_n;
  
  const int threads_per_block = 256;
  const int blocks_per_grid = (data_n - 1) / threads_per_block + 1;

  std::vector<uint8_t> h_data_in(data_n * data_size);
  std::vector<double> h_data_out(data_n);

  for(auto &i : h_data_in) {
    i = rand() % 256;
  }

  uint8_t *d_data_in;
  double *d_data_out;
  cudaMalloc(&d_data_in, data_n * data_size);
  cudaMalloc(&d_data_out, data_n * sizeof(double));

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_data_in, h_data_in.data(), data_n * data_size, cudaMemcpyHostToDevice);

  vectorSum<<<blocks_per_grid, threads_per_block>>>(d_data_in, d_data_out, 31, data_n, data_size);

  cudaMemcpy(h_data_out.data(), d_data_out, data_n, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 10; i++) {
    std::cout << h_data_out[i] << std::endl;
    int offset = i * data_size;
    auto start = h_data_in.begin() + offset;
    auto end = h_data_in.begin() + offset + data_size;

    std::vector<uint8_t> v(start, end);
    auto chi = lc_test(v, 31);
    std::cout << chi << std::endl << std::endl;
  }

  // Free memory on device
  cudaFree(d_data_in);
  cudaFree(d_data_out);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}