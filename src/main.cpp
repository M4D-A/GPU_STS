#include "io.hpp"
#include "linear_complexity.hpp"
#include <vector>

int main(int argc, char *argv[]){
    std::vector<uint8_t> data = read_file("data.dat");
    long double chi = lc_test(data, 7);
    std::cout << "Chi: " << chi << std::endl;
}
