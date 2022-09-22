#include "mr_kernel.hu"
#include "io.hpp"

#ifdef __CUDA_ARCH__
#define __const__  __constant__ const
#else
#define __const__ const
#endif

/// DATA /// 
__const__ double rank_probabilities[65][3] = {
    {1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000},
    {0.50000000000000000000, 0.50000000000000000000, 0.00000000000000000000},
    {0.37500000000000000000, 0.56250000000000000000, 0.06250000000000000000},
    {0.32812500000000000000, 0.57421875000000000000, 0.09765625000000000000},
    {0.30761718750000000000, 0.57678222656250000000, 0.11560058593750000000},
    {0.29800415039062500000, 0.57738304138183593761, 0.12461280822753906239},
    {0.29334783554077148438, 0.57752855122089385975, 0.12912361323833465587},
    {0.29105605557560920715, 0.57756436028284952044, 0.13137958414154127240},
    {0.28991911785851698369, 0.57757324260876430333, 0.13250763953271871298},
    {0.28935286958144956770, 0.57757545451609659793, 0.13307167590245383437},
    {0.28907029841974893336, 0.57757600641289679459, 0.13335369516735427205},
    {0.28892915081309866531, 0.57757614425235641396, 0.13349470493454492073},
    {0.28885861146963843617, 0.57757617869539521296, 0.13356520983496635089},
    {0.28882335040866802131, 0.57757618730405267638, 0.13360046228727930230},
    {0.28880572203034717975, 0.57757618945595432685, 0.13361808851369849340},
    {0.28879690837916217156, 0.57757618999389690416, 0.13362690162694092428},
    {0.28879250168805531182, 0.57757619012837844411, 0.13363130818356624410},
    {0.28879029837612226698, 0.57757619016199831624, 0.13363351146187941678},
    {0.28878919672856071256, 0.57757619017040322024, 0.13363461310103606718},
    {0.28878864590688116130, 0.57757619017250443820, 0.13363516392061440053},
    {0.28878837049656669019, 0.57757619017302974163, 0.13363543933040356818},
    {0.28878823279154078047, 0.57757619017316106741, 0.13363557703529815212},
    {0.28878816393906065708, 0.57757619017319389884, 0.13363564588774544406},
    {0.28878812951282880321, 0.57757619017320210658, 0.13363568031396909022},
    {0.28878811229971492825, 0.57757619017320415864, 0.13363569752708091311},
    {0.28878810369315850376, 0.57757619017320467158, 0.13363570613363682463},
    {0.28878809938988041981, 0.57757619017320479979, 0.13363571043691478040},
    {0.28878809723824140984, 0.57757619017320483172, 0.13363571258855375844},
    {0.28878809616242191288, 0.57757619017320483990, 0.13363571366437324722},
    {0.28878809562451216640, 0.57757619017320484191, 0.13363571420228299169},
    {0.28878809535555729365, 0.57757619017320484223, 0.13363571447123786412},
    {0.28878809522107985744, 0.57757619017320484267, 0.13363571460571529992},
    {0.28878809515384113935, 0.57757619017320484267, 0.13363571467295401799},
    {0.28878809512022178029, 0.57757619017320484267, 0.13363571470657337702},
    {0.28878809510341210074, 0.57757619017320484267, 0.13363571472338305659},
    {0.28878809509500726104, 0.57757619017320484267, 0.13363571473178789629},
    {0.28878809509080484113, 0.57757619017320484267, 0.13363571473599031617},
    {0.28878809508870363122, 0.57757619017320484267, 0.13363571473809152611},
    {0.28878809508765302622, 0.57757619017320484267, 0.13363571473914213114},
    {0.28878809508712772374, 0.57757619017320484267, 0.13363571473966743359},
    {0.28878809508686507251, 0.57757619017320484267, 0.13363571473993008482},
    {0.28878809508673374689, 0.57757619017320484267, 0.13363571474006141044},
    {0.28878809508666808409, 0.57757619017320484267, 0.13363571474012707325},
    {0.28878809508663525268, 0.57757619017320484267, 0.13363571474015990468},
    {0.28878809508661883700, 0.57757619017320484267, 0.13363571474017632036},
    {0.28878809508661062912, 0.57757619017320484267, 0.13363571474018452821},
    {0.28878809508660652523, 0.57757619017320484267, 0.13363571474018863213},
    {0.28878809508660447324, 0.57757619017320484267, 0.13363571474019068409},
    {0.28878809508660344726, 0.57757619017320484267, 0.13363571474019171007},
    {0.28878809508660293427, 0.57757619017320484267, 0.13363571474019222306},
    {0.28878809508660267777, 0.57757619017320484267, 0.13363571474019247953},
    {0.28878809508660254954, 0.57757619017320484267, 0.13363571474019260779},
    {0.28878809508660248541, 0.57757619017320484267, 0.13363571474019267192},
    {0.28878809508660245335, 0.57757619017320484267, 0.13363571474019270396},
    {0.28878809508660243733, 0.57757619017320484267, 0.13363571474019272001},
    {0.28878809508660242930, 0.57757619017320484267, 0.13363571474019272803},
    {0.28878809508660242526, 0.57757619017320484267, 0.13363571474019273204},
    {0.28878809508660242329, 0.57757619017320484267, 0.13363571474019273405},
    {0.28878809508660242226, 0.57757619017320484267, 0.13363571474019273508},
    {0.28878809508660242177, 0.57757619017320484267, 0.13363571474019273556},
    {0.28878809508660242155, 0.57757619017320484267, 0.13363571474019273578},
    {0.28878809508660242142, 0.57757619017320484267, 0.13363571474019273594},
    {0.28878809508660242136, 0.57757619017320484267, 0.13363571474019273594},
    {0.28878809508660242131, 0.57757619017320484267, 0.13363571474019273605},
    {0.28878809508660242128, 0.57757619017320484267, 0.13363571474019273605}
};


/// FUNCTIONS ///
__host__ __device__ uint64_t extract_bits_mr(uint8_t *data, uint32_t start, uint32_t bits) {
    uint64_t result = 0;
    uint32_t byte = start / 8;
    uint32_t bit = start % 8;
    uint32_t bits_left = bits;
    while (bits_left > 0) {
        uint32_t bits_to_read = (8 - bit < bits_left) ? 8 - bit : bits_left;
        uint64_t mask = (1 << bits_to_read) - 1;
        result = (result << bits_to_read) | ((data[byte] >> (8 - bit - bits_to_read)) & mask);
        bits_left -= bits_to_read;
        bit = 0;
        byte++;
    }
    return result;
}

__host__ __device__ uint64_t matrix_rank(uint64_t *matrix, uint32_t max_rank) { 
    int64_t rank = 0;
    int64_t i, j;
    int64_t temp_rank;
    uint64_t row, mask;

    for (i = 0; i < max_rank; ++i) {
        mask = 1UL << i;
        temp_rank = -1;

        for (j = rank; j < max_rank; ++j) {
            temp_rank = (matrix[j] & mask && temp_rank == -1) ? j : temp_rank;
        }

        uint64_t condition = ((temp_rank != -1) && (rank != temp_rank));
        row = (condition) ? matrix[rank] : row;
        matrix[rank] = (condition) ? matrix[temp_rank] : matrix[rank];
        matrix[temp_rank] = (condition) ? row : matrix[temp_rank];

        for (j = rank + 1; j < max_rank; ++j) {
            matrix[j] ^= (matrix[j] & mask && temp_rank != -1) ? matrix[rank] : 0;
        }
        rank += (temp_rank != -1) ? 1 : 0;
    }
    return rank;
}

__host__ __device__ double mr_test(uint8_t *data, uint32_t data_len, uint32_t max_rank) {

    uint32_t bit_data_len = data_len * 8;
    uint32_t matrices_num = bit_data_len / (max_rank * max_rank);
    uint64_t matrix[64];
    uint32_t occurrences[3] = {0, 0, 0};
    uint32_t i, j;
    for (i = 0; i < matrices_num; i++) {
        for (j = 0; j < max_rank; j++) {
            uint32_t starting_bit = (i * max_rank * max_rank) + (j * max_rank);
            uint64_t row = extract_bits_mr(data, starting_bit, max_rank);
            matrix[j] = row;
        }

        uint32_t rank = matrix_rank(matrix, max_rank); /// [KERNELIZED]
        uint32_t rank_diff = (max_rank - rank > 2) ? 2 : max_rank - rank;
        occurrences[rank_diff]++;
    }

    double chi = 0.0;
    for (i = 0; i < 3; i++) {
        double expected = matrices_num * rank_probabilities[max_rank][i];
        chi += (occurrences[i] - expected) * (occurrences[i] - expected) / expected;
    }
    return chi;
}

/// KERNEL
__global__ void mr_kernel(
    const uint8_t *data,
    uint32_t data_num, 
    uint32_t data_size,
    uint32_t bits,
    double*__restrict data_out) {

    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id < data_num){
        uint64_t byte_offset = (uint64_t)thread_id * (uint64_t)data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[thread_id] = mr_test(thred_data, data_size, bits);
    }
}

/// GPU LAUNCHER
std::vector<double> run_gpu_mr_tests(
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

    mr_kernel<<<blocks_per_grid, threads_per_block>>>(dev_data_in, data_num, data_size, bits, dev_data_out);

    std::vector<double> data_out(data_num);
    cudaMemcpy(data_out.data(), dev_data_out, data_num * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_data_in);
    cudaFree(dev_data_out);

    return data_out;
}


/// CPU LAUNCHER
std::vector<double> run_cpu_mr_tests(
    const std::vector<uint8_t> &data,
    uint64_t data_num,
    uint64_t data_size,
    uint64_t bits) {
    std::vector<double> data_out(data_num);
    for (uint64_t i = 0; i < data_num; i++) {
        uint64_t byte_offset = i * data_size;
        uint8_t* thred_data = (uint8_t*)&data[byte_offset];
        data_out[i] = mr_test(thred_data, data_size, bits);
    }
    return data_out;
}


/// PERFORMANCE TESTS
int mr_perf(uint64_t data_num, uint64_t data_size, uint64_t bits) {
    float gb = ((float)data_num * (float)data_size / (float)(1<<30));
    printf("MR-%lu test on %lu samples of size %luB (%lfGB of data)\n", bits, data_num, data_size, gb);
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
    run_gpu_mr_tests(data_pieces, data_num, data_size, bits); // warmup
    cudaEvent_t gpu_start, gpu_stop;
    float gpu_time, gpu_throughput;
    cudaEventCreate(&gpu_start);
    cudaEventRecord(gpu_start, 0);

    std::vector<double> gpu_out = run_gpu_mr_tests(data_pieces, data_num, data_size, bits);

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

    std::vector<double> cpu_out = run_cpu_mr_tests(data_pieces, data_num, data_size, bits);

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

