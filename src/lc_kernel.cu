#include "lc_kernel.hu"
#include "io.hpp"
#ifdef __CUDA_ARCH__
#define __const__  __constant__ const
#else
#define __const__ const
#endif


/// DATA /// 
__const__ double probs[7] = {0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833};
__const__ uint8_t byterev[256] = { 
0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0, 
0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 
0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4, 
0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc, 
0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 
0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa, 
0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6, 
0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 
0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1, 
0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9, 
0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 
0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd, 
0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3, 
0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 
0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7, 
0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff, 
};


/// HOST-DEVICE FUNCTIONS
__host__ __device__ uint64_t extract_reverse_bits(uint8_t* data, uint64_t start, uint64_t bits){
    uint64_t end = start + bits; // [CHECK] -1?

    uint64_t s_byte = start / 8;
    uint64_t s_bit = start % 8;

    uint64_t e_byte = end / 8;
    uint64_t e_bit = end % 8;

    uint8_t right_bit_mask = (1 << (e_bit)) - 1;

    uint64_t output = 0;
    uint64_t i;
    for (i = e_byte; i >= s_byte; i--){
        uint8_t current_byte = byterev[data[i]];
        current_byte &= (i == e_byte) ? right_bit_mask : 0xff;
        current_byte >>= (i == s_byte) ? (s_bit) : 0;
        output <<= (i == s_byte) ? (8-s_bit) : 8;
        output |= current_byte;
        if(i == s_byte) break;
    }
    return output;
}

__host__ __device__ uint64_t parity(uint64_t num) {
    #ifdef  __CUDA_ARCH__
        return (__popcll(num) & 1);
    #else
        return __builtin_parityll(num);
    #endif
}

__host__ __device__ uint64_t trailing_zeros(uint64_t num) {
    #ifdef  __CUDA_ARCH__
        return __ffsll(num) - 1;
    #else
        return __builtin_ctzll(num);
    #endif
}

__host__ __device__ uint64_t complexity(uint64_t sequence, uint64_t bits) {
    if(sequence == 0){
        return 1;
    }
    uint64_t N = bits;
    uint64_t k = trailing_zeros(sequence);
    uint64_t F = (1 << (k + 1)) | 1;
    uint64_t G = 1;
    uint64_t l = k + 1;
    uint64_t a = k;
    uint64_t b = 0;
    uint64_t n = k;

    for(n = k + 1; n < N; n++){
        uint64_t d = parity(sequence & (F << (n - l)));
        if(d == 0){
            b+=1;
        }
        else{
            if(2*l > n){
                F ^= (G << (a - b));
                b += 1;
            }
            else{
                uint64_t T = F;
                F = (F << (b - a)) ^ G;
                l = n + 1 - l;
                G = T;
                a = b;
                b = n - l + 1;
            }
        }
    }

    return l;
}

__host__ __device__ double lc_test(uint8_t* data, uint64_t data_size, uint64_t bits) {
    uint64_t bins[7] = {0, 0, 0, 0, 0, 0, 0};
    uint64_t sequences_num = (data_size * 8) / bits;
    uint64_t i;

    double s_one = (bits & 1) ? -1.0 : 1.0;
    double mi = (double) (bits / 2.0);
    mi += (9.0 - s_one) / 36.0;
    mi -= ((bits / 3.0) + (2.0 / 9.0)) / pow(2.0, bits);

    for (i = 0; i < sequences_num; i++) {
        uint64_t starting_bit = i * bits;
        
        uint64_t sequence = extract_reverse_bits(data, starting_bit, bits);
        uint64_t lc = complexity(sequence, bits);

        double ti = s_one * ((double) lc - mi) + 2.0 / 9.0;
        bins[0] += (ti <= -2.5) ? 1u : 0;
        bins[1] += (ti > -2.5 && ti <= -1.5) ? 1u : 0;
        bins[2] += (ti > -1.5 && ti <= -0.5) ? 1u : 0;
        bins[3] += (ti > -0.5 && ti <= 0.5) ? 1u : 0;
        bins[4] += (ti > 0.5 && ti <= 1.5) ? 1u : 0;
        bins[5] += (ti > 1.5 && ti <= 2.5) ? 1u : 0;
        bins[6] += (ti > 2.5) ? 1u : 0;
    }

    double chi = 0.0;
    for (i = 0; i < 7; i++) {
        double expected = probs[i] * sequences_num;
        double enumerator = pow((double) bins[i] - expected, 2.0);
        chi += enumerator / expected;
    }
    
    return chi;
    
}


/// KERNEL
__global__ void lc_kernel(
    const uint8_t *data,
    uint64_t data_num, 
    uint64_t data_size,
    uint64_t bits,
    double*__restrict data_out,
    uint64_t thread_offset) {

    uint64_t thread_id = thread_offset + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id < data_num){
        uint64_t byte_offset = thread_id * data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[thread_id] = lc_test(thred_data, data_size, bits);
    }
}


/// GPU LAUNCHER
std::vector<double> run_gpu_lc_tests(
    const std::vector<uint8_t> &data,
    uint64_t data_num,
    uint64_t data_size,
    uint64_t bits) {

    uint64_t threads_per_block = 256;
    uint64_t blocks_per_grid = (data_num + threads_per_block - 1) / threads_per_block;

    uint8_t* dev_data_in;
    double* dev_data_out;
    cudaMalloc((void**)&dev_data_in, data_num * data_size * sizeof(uint8_t));
    cudaMalloc((void**)&dev_data_out, data_num * sizeof(double));

    cudaMemcpy(dev_data_in, data.data(), data_num * data_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    lc_kernel<<<blocks_per_grid, threads_per_block>>>(dev_data_in, data_num, data_size, bits, dev_data_out);

    std::vector<double> data_out(data_num);
    cudaMemcpy(data_out.data(), dev_data_out, data_num * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_data_in);
    cudaFree(dev_data_out);

    return data_out;
}


/// CPU LAUNCHER
std::vector<double> run_cpu_lc_tests(
    const std::vector<uint8_t> &data,
    uint64_t data_num,
    uint64_t data_size,
    uint64_t bits) {
    std::vector<double> data_out(data_num);
    for (uint64_t i = 0; i < data_num; i++) {
        uint64_t byte_offset = i * data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[i] = lc_test(thred_data, data_size, bits);
    }
    return data_out;
}


/// PERFORMANCE TESTS
int lc_perf(uint64_t data_num, uint64_t data_size, uint64_t bits) {

    // DATA GENERATION
    cudaEvent_t gen_start, gen_stop;
    float gen_time;
    cudaEventCreate(&gen_start);
    cudaEventRecord(gen_start, 0);

    std::vector<uint8_t> data_pieces(data_num * data_size);
    uint8_t r = 117;
    for (uint64_t i = 0; i < data_num * data_size; i++) {
        r = (r * r) + 117 * r + 17 + i;
        data_pieces[i] = r;
    }

    cudaEventCreate(&gen_stop);
    cudaEventRecord(gen_stop, 0);
    cudaEventSynchronize(gen_stop);
    cudaEventElapsedTime(&gen_time, gen_start, gen_stop);
    std::cout << "Data generation time: " << gen_time << " ms" << std::endl;

    // GPU TEST
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_time;
    cudaEventCreate(&gpu_start);
    cudaEventRecord(gpu_start, 0);

    std::vector<double> gpu_out = run_gpu_lc_tests(data_pieces, data_num, data_size, bits);

    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

    // CPU TEST
    cudaEvent_t cpu_start, cpu_stop;
    float cpu_time;
    cudaEventCreate(&cpu_start);
    cudaEventRecord(cpu_start, 0);

    std::vector<double> cpu_out = run_cpu_lc_tests(data_pieces, data_num, data_size, bits);

    cudaEventCreate(&cpu_stop);
    cudaEventRecord(cpu_stop, 0);
    cudaEventSynchronize(cpu_stop);
    cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

    // CHECK
    for (uint64_t i = 0; i < data_num; i++) {
        if (abs(gpu_out[i] - cpu_out[i]) > 0.0001){
            std::cout << "ERROR: " << i << " " << gpu_out[i] << " " << cpu_out[i] << std::endl;
        }
    }

    std::cout << "GPU speedup: " << cpu_time / gpu_time << std::endl;
    return 0;
}
