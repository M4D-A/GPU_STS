#include "io.hpp"
#include "lc_kernel.hu"
#include <iostream>
#include <vector>
#include <assert.h>


int main(int argc, char **argv){
  auto data = read_file(argv[1]);
  auto n = lc_test(data.data(), data.size(), atoi(argv[2]));
  printf("%lf\n",n);
  return 0;
}