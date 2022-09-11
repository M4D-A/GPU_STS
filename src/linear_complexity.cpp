#include "linear_complexity.hpp"
#include "io.hpp"
#include <math.h>

long double probs[7] = {0.01047L, 0.03125L, 0.12500L, 0.50000L, 0.25000L, 0.06250L, 0.020833L};

uint64_t complexity(uint64_t sequence, uint64_t length) {
    if(sequence == 0){
        return 1;
    }
    uint64_t N = length;

    uint64_t k = trailing_zeros(sequence);
    uint64_t F = (1 << (k + 1)) | 1;
    uint64_t G = 1;
    uint64_t l = k + 1;
    uint64_t a = k;
    uint64_t b = 0;
    uint64_t n = k;

    for(n = k + 1; n < N; n++){
        uint64_t d = uintxor(sequence & (F << (n - l)));
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
                b = n - l +1;
            }
        }
    }

    return l;
}

long double lc_test(std::vector<uint8_t> data, uint64_t bit_sequence_len) {
    uint64_t bins[7] = {0, 0, 0, 0, 0, 0, 0};
    uint64_t sequences_num = (data.size() * 8) / bit_sequence_len;
    uint64_t i;

    long double s_one = (bit_sequence_len & 1) ? -1.0 : 1.0;
    long double mi = (long double) (bit_sequence_len >> 1); /// ?
    mi += (9.0L - s_one) / 36.0L;
    mi -= ((bit_sequence_len / 3.0L) + (2.0L / 9.0L)) / powl(2.0L, bit_sequence_len);

    for (i = 0; i < sequences_num; i++) {
        uint64_t starting_bit = i * bit_sequence_len;
        uint64_t sequence = extract_bits(data, starting_bit, bit_sequence_len);
        sequence = reverse_uint64_t(sequence, bit_sequence_len); /// [KERNELIZED]
        uint64_t lc = complexity(sequence, bit_sequence_len);

        long double ti = s_one * ((long double) lc - mi) + 2.0 / 9.0L;
        bins[0] += (ti <= -2.5L) ? 1u : 0;
        bins[1] += (ti > -2.5L && ti <= -1.5L) ? 1u : 0;
        bins[2] += (ti > -1.5L && ti <= -0.5L) ? 1u : 0;
        bins[3] += (ti > -0.5L && ti <= 0.5L) ? 1u : 0;
        bins[4] += (ti > 0.5L && ti <= 1.5L) ? 1u : 0;
        bins[5] += (ti > 1.5L && ti <= 2.5L) ? 1u : 0;
        bins[6] += (ti > 2.5L && ti) ? 1u : 0;
    }

    long double chi = 0.0L;
    for (i = 0; i < 7; i++) {
        long double expected = probs[i] * sequences_num;
        long double enumerator = powl((long double) bins[i] - expected, 2.0L);
        chi += enumerator / expected;
    }
    return chi;
}