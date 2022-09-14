#include "linear_complexity.hpp"
#include "io.hpp"
#include "lc_kernel.hu"
#include <iostream>
#include <vector>
#include <assert.h>


int main() {
  int files = 1024 * 1024; // 64k files
  int file_size = 1024; // 16KB = 128Kbits


  // DATA GENERATION
  cudaEvent_t gen_start, gen_stop;
  float gen_time;

  cudaEventCreate(&gen_start);
  cudaEventRecord(gen_start, 0);

  std::vector<std::vector<uint8_t>> data_pieces(files);
  for(auto &piece : data_pieces) {
    piece.resize(file_size);
    for(int i = 0; i < file_size; i++) {
      piece[i] = rand() % 256;
    }
  }

  std::vector<uint8_t> data;
  for(auto &piece : data_pieces) {
    data.insert(data.end(), piece.begin(), piece.end());
  }

  cudaEventCreate(&gen_stop);
  cudaEventRecord(gen_stop, 0);
  cudaEventSynchronize(gen_stop);
  cudaEventElapsedTime(&gen_time, gen_start, gen_stop);
  std::cout << "Data generation time: " << gen_time << " ms" << std::endl;
  

  std::vector<double> dev_results(files);
  std::vector<double> host_results(files);

  // GPU
  cudaEvent_t dev_start, dev_stop;
  float dev_elapsedTime;

  cudaEventCreate(&dev_start);
  cudaEventRecord(dev_start,0);

  run_lc_tests(data.data(), files, file_size, 31, dev_results.data());

  cudaEventCreate(&dev_stop);
  cudaEventRecord(dev_stop,0);
  cudaEventSynchronize(dev_stop);

  cudaEventElapsedTime(&dev_elapsedTime, dev_start,dev_stop);
  printf("Device time: %f ms\n", dev_elapsedTime);

  // CPU
  cudaEvent_t host_start, host_stop;
  float host_elapsedTime;

  cudaEventCreate(&host_start);
  cudaEventRecord(host_start,0);

  for(int i = 0; i < files; i++) {
    host_results[i] = lc_test(data_pieces[i], 31);
  }

  cudaEventCreate(&host_stop);
  cudaEventRecord(host_stop,0);
  cudaEventSynchronize(host_stop);

  cudaEventElapsedTime(&host_elapsedTime, host_start,host_stop);
  printf("Host time: %f ms\n", host_elapsedTime);

  // Compare
  for(int i = 0; i < files; i++) {
    assert(abs(dev_results[i] - host_results[i]) < 0.0001);
  }

  printf("Success!\n");

  return 0;
}