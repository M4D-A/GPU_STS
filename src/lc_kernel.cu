#include "lc_kernel.hu"
#include "io.hpp"

#ifdef __CUDA_ARCH__
#define __const__  __constant__ const
#else
#define __const__ const
#endif


/// DATA /// 
__const__ double probs[7] = {0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833};

/// HOST-DEVICE FUNCTIONS
__host__ __device__ uint64_t extract_bits(uint8_t *data, uint32_t start, uint32_t bits) {
    uint64_t result = 0;
    uint32_t byte = start / 8;
    uint32_t bit = start % 8;
    uint32_t bits_left = bits;
    while (bits_left > 0) {
        uint32_t bits_to_read = min(8 - bit, bits_left);
        uint64_t mask = (1 << bits_to_read) - 1;
        result = (result << bits_to_read) | ((data[byte] >> (8 - bit - bits_to_read)) & mask);
        bits_left -= bits_to_read;
        bit = 0;
        byte++;
    }
    return result;
}

__host__ __device__ uint32_t parity(uint64_t sequence) {
    #ifdef  __CUDA_ARCH__
        return (__popcll(sequence) & 1);
    #else
        return __builtin_parityll(sequence);
    #endif
}

__host__ __device__ uint32_t leading_zeros(uint64_t sequence) {
    #ifdef  __CUDA_ARCH__
        return __clzll(sequence);
    #else
        return __builtin_clzll(sequence);
    #endif
}

__host__ __device__ uint32_t complexity(uint64_t sequence, uint32_t bits) {
    if(sequence == 0){
        return 1;
    }
    sequence <<= (64 - bits);
    uint32_t k = leading_zeros(sequence);
    uint64_t F = (1ull << 63) | (1ull << (62 - k));
    uint64_t G = (1ull << 63);
    uint32_t l = k + 1;
    uint32_t a = k;
    uint32_t b = 0;

    for(k = k + 1; k < bits; k++){
        uint32_t d = parity(sequence & (F >> (k - l)));
        if(d == 0){
            b+=1;
        }
        else{
            if(2*l > k){
                F ^= (G >> (a - b));
                b += 1;
            }
            else{
                uint64_t T = F;
                F = (F >> (b - a)) ^ G;
                G = T;


                l = k + 1 - l;
                a = b;
                b = k - l + 1;
            }
        }
    }

    return l;
}

__host__ __device__ double lc_test(uint8_t* data, uint32_t data_size, uint32_t bits) {
    uint32_t bins[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t sequences_num = (data_size * 8) / bits;
    uint32_t i;

    double s_one = (bits & 1) ? -1.0 : 1.0;
    double mi = (double) (bits / 2.0);
    mi += (9.0 - s_one) / 36.0;
    mi -= ((bits / 3.0) + (2.0/ 9.0)) / pow(2.0, bits);

    for (i = 0; i < sequences_num; i++) {
        uint32_t starting_bit = i * bits;
        
        uint64_t sequence = extract_bits(data, starting_bit, bits);
        uint32_t lc = complexity(sequence, bits);

        double ti = s_one * ((double) lc - mi) + 2.0/ 9.0;
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
    uint32_t data_num, 
    uint32_t data_size,
    uint32_t bits,
    double*__restrict data_out) {

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id < data_num){
        uint64_t byte_offset = (uint64_t)thread_id * (uint64_t)data_size;
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
    float gb = ((float)data_num * (float)data_size / (float)(1<<30));
    printf("LC-%lu test on %lu samples of size %luB (%lfGB)\n", bits, data_num, data_size, gb);
    // DATA GENERATION
    cudaEvent_t gen_start, gen_stop;
    float gen_time, gen_throughput;
    cudaEventCreate(&gen_start);
    cudaEventRecord(gen_start, 0);

    std::vector<uint8_t> data_pieces(data_num * data_size);
    //uint8_t r = 117;
    for (uint64_t i = 0; i < data_num * data_size; i++) {
        //r = (r * r) + 117 * r + 17 + i;
        data_pieces[i] = rand() % 256;
    }

    cudaEventCreate(&gen_stop);
    cudaEventRecord(gen_stop, 0);
    cudaEventSynchronize(gen_stop);
    cudaEventElapsedTime(&gen_time, gen_start, gen_stop);
    gen_throughput = gb / (gen_time / 1000);
    printf("Data generation time: %f ms, throughput: %f GB/s\n", gen_time, gen_throughput);

    // GPU TEST
    run_gpu_lc_tests(data_pieces, data_num, data_size, bits); // warmup
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_time, gpu_throughput;
    cudaEventCreate(&gpu_start);
    cudaEventRecord(gpu_start, 0);

    std::vector<double> gpu_out = run_gpu_lc_tests(data_pieces, data_num, data_size, bits);

    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    gpu_throughput = gb / (gpu_time / 1000);
    printf("GPU time: %f ms, throughput: %f GB/s\n", gpu_time, gpu_throughput);


    // CPU TEST
    cudaEvent_t cpu_start, cpu_stop;
    float cpu_time, cpu_throughput;
    cudaEventCreate(&cpu_start);
    cudaEventRecord(cpu_start, 0);

    std::vector<double> cpu_out = run_cpu_lc_tests(data_pieces, data_num, data_size, bits);

    cudaEventCreate(&cpu_stop);
    cudaEventRecord(cpu_stop, 0);
    cudaEventSynchronize(cpu_stop);
    cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);
    cpu_throughput = gb / (cpu_time / 1000);
    printf("CPU time: %f ms, throughput: %f GB/s\n", cpu_time, cpu_throughput);


    // CHECK
    for (uint64_t i = 0; i < data_num; i++) {
        if (abs(gpu_out[i] - cpu_out[i]) > 1e-6) {
            std::cout << "ERROR: " << i << " " << gpu_out[i] << " " << cpu_out[i] << std::endl;
        }
    }

    std::cout << "GPU speedup: " << cpu_time / gpu_time << std::endl;
    printf("\n");
    return 0;
}
