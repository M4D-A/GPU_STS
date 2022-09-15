#include "linear_complexity.hpp"
#include "io.hpp"
#include "lc_kernel.hu"
#include <iostream>
#include <vector>
#include <assert.h>


int main() {
  for(int fsize = 4096; fsize <= 1024*1024; fsize*=2){
    printf("fsize: %d\n", fsize);
    for(int fnum = 32; fnum <= 64*1024; fnum*=2){
      printf("fnum: %d\n", fnum);
      lc_test(fnum, fsize ,31);
      printf("\n");
    }
    printf("###############\n");
  }
  return 0;
}