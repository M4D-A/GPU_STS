#include "lr_kernel.hu"

#ifdef __CUDA_ARCH__
    #define __const__  __constant__ const
#else
    #define __const__ const
#endif

/// DATA ///
__const__ uint64_t BIT_LEN[5] = {8, 128, 512, 1000, 10000};
__const__ uint64_t BIN_NUM[5] = {4, 6, 6, 6, 7};
__const__ uint64_t MIN_BIN[5] = {1, 4, 6, 7, 10};
__const__ uint64_t MAX_BIN[5] = {4, 9, 11, 12, 16};
__const__ double BIN_PROBS[5][7] = {
    {0.21484375,             0.3671875,              0.23046875,             0.1875,                 0.0,                    0.0,         0.0},
    {0.11740357883779323143, 0.24295595927745485921, 0.24936348317907796964, 0.17517706034678234949, 0.10270107130405369391, 0.112398847, 0.0},
    {0.12993348330772888835, 0.23612234530786038173, 0.24183437193063447209, 0.17297542562939206267, 0.10326976304604072372, 0.115864611, 0.0},
    {0.13885519087000345173, 0.23690381667371765447, 0.23879124004379725789, 0.17001807921942845119, 0.10143614403657023491, 0.113995529, 0.0},
    {0.0882,                 0.2092,                 0.2483,                 0.1933,                 0.1208,                 0.0675,      0.0727}
};
__const__ uint8_t right_run[256] = {
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8,
};
__const__ uint8_t left_run[256] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8,
};
__const__ uint8_t longest_run[256] = {
        0, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 4,
        1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 2, 2, 3, 3, 4, 5,
        1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 4,
        2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6,
        1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 4,
        1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 2, 2, 3, 3, 4, 5,
        2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 4,
        3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7,
        1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 4,
        1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 2, 2, 3, 3, 4, 5,
        1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 4,
        2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6,
        2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 4,
        2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 4, 5,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8,
};

/// MAIN ///
__host__ __device__ uint64_t sequence_longest_run(const uint8_t *sequence, uint64_t seq_len) {
    uint64_t seq_longest_run = 0;
    uint64_t current_longest_run = 0;
    uint64_t temp;
    uint64_t i;
    for (i = 0; i < seq_len; i++) {
        uint8_t byte_value = sequence[i];

        uint8_t byte_left_run = left_run[byte_value];
        uint8_t byte_right_run = right_run[byte_value];
        uint8_t byte_longest_run = longest_run[byte_value];

        seq_longest_run = (byte_longest_run > seq_longest_run) ? byte_longest_run : seq_longest_run;

        temp = current_longest_run + byte_left_run;
        seq_longest_run = (temp > seq_longest_run) ? temp : seq_longest_run;

        current_longest_run = (byte_value == 0xff) ? current_longest_run + 8 : byte_right_run;
    }
    return seq_longest_run;
}

__host__ __device__ double lr_test(const uint8_t *data, uint64_t data_len, uint64_t mode) {
    uint64_t bit_sequence_len = BIT_LEN[mode];
    uint64_t min_bin = MIN_BIN[mode];
    uint64_t max_bin = MAX_BIN[mode];
    uint64_t bin_num = BIN_NUM[mode];
    const double *bin_probs = BIN_PROBS[mode];

    uint64_t sequence_len = bit_sequence_len / 8;
    uint64_t seq_num = data_len / sequence_len;
    uint64_t i;
    uint64_t occurrences[16];
    for (i = 0; i < 16; i++) {
        occurrences[i] = 0;
    }

    uint64_t offset;
    for (offset = 0; offset + sequence_len <= data_len; offset += sequence_len) {
        const uint8_t *sequence_start = data + offset;
        uint64_t run = sequence_longest_run(sequence_start, sequence_len); ///[KERNELIZED]
        uint64_t bin_index = (run < min_bin) ? 0 :
                             (run > max_bin) ? bin_num - 1 : run - min_bin;
        occurrences[bin_index]++;
    }
    double chi = 0.0;
    for (i = 0; i < bin_num; i++) {
        double expected = bin_probs[i] * (double) seq_num;
        double l = (double) occurrences[i] - expected;
        double l2 = l * l;
        chi += l2 / expected;
    }
    return chi;
}

/// KERNEL
__global__ void lr_kernel(
    const uint8_t *data,
    uint32_t data_num, 
    uint32_t data_size,
    uint32_t mode,
    double*__restrict data_out) {   


    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id < data_num){
        uint64_t byte_offset = (uint64_t)thread_id * (uint64_t)data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[thread_id] = lr_test(thred_data, data_size, mode);
    }
}

std::vector<double> run_gpu_lr_tests(
    const std::vector<uint8_t> &data,
    uint64_t data_num,
    uint64_t data_size,
    uint64_t mode) {

    uint64_t threads_per_block = 256;
    uint64_t blocks_per_grid = (data_num + threads_per_block - 1) / threads_per_block;

    uint8_t* dev_data_in;
    double* dev_data_out;
    cudaMalloc((void**)&dev_data_in, data_num * data_size * sizeof(uint8_t));
    cudaMalloc((void**)&dev_data_out, data_num * sizeof(double));

    cudaMemcpy(dev_data_in, data.data(), data_num * data_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    lr_kernel<<<blocks_per_grid, threads_per_block>>>(dev_data_in, data_num, data_size, mode, dev_data_out);

    std::vector<double> data_out(data_num);
    cudaMemcpy(data_out.data(), dev_data_out, data_num * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_data_in);
    cudaFree(dev_data_out);

    return data_out;
}

std::vector<double> run_cpu_lr_tests(
    const std::vector<uint8_t> &data,
    uint64_t data_num,
    uint64_t data_size,
    uint64_t mode) {

    std::vector<double> data_out(data_num);
    for (uint64_t i = 0; i < data_num; i++) {
        uint64_t byte_offset = i * data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[i] = lr_test(thred_data, data_size, mode);
    }
    return data_out;
}

int lr_perf(uint64_t data_num, uint64_t data_size, uint64_t mode) {
    float gb = ((float)data_num * (float)data_size / (float)(1<<30));
    printf("LR-%lu test on %lu samples of size %luB (%lfGB)\n", mode, data_num, data_size, gb);
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
    run_gpu_lr_tests(data_pieces, data_num, data_size, mode); // warmup
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_time, gpu_throughput;
    cudaEventCreate(&gpu_start);
    cudaEventRecord(gpu_start, 0);

    std::vector<double> gpu_out = run_gpu_lr_tests(data_pieces, data_num, data_size, mode);

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

    std::vector<double> cpu_out = run_cpu_lr_tests(data_pieces, data_num, data_size, mode);

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