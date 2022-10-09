#include "io.hpp"
#include "lc_kernel.hu"
#include "lr_kernel.hu"
#include "mr_kernel.hu"
#include "cephes.hpp"

int main(int argc, char **argv){
  lc_perf(32*1024, 1024, 31);
}