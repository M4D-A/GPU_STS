#include "io.hpp"
#include "lc_kernel.hu"
#include <iostream>
#include <vector>
#include <assert.h>


int main(int argc, char **argv){
  lc_perf(1204*1024, 1024, 32);
  return 0;
}