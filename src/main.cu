#include "io.hpp"
#include "lc_kernel.hu"
#include "mr_kernel.hu"

int main(int argc, char **argv){
  mr_perf(32*1024, 1024, 32);
  lc_perf(32*1024, 1024, 32);
  return 0;
}