/* 最小性能探针：只调用一次 conv2d，供 perf stat 采集干净的硬件计数器。
 *
 * 目的：bench_conv.c 内部含预热、参考实现与校验，perf 统计会混入无关代码。
 * 本探针把被测代码隔离成单次调用，让 cycles / instructions 可归因。
 *
 * 用法:
 *   gcc -O3 -fopenmp -march=native -ffp-contract=off \
 *       tools/perf_probe.c src/conv2d_v7.c -lm -o build/probe_v7
 *   OMP_NUM_THREADS=6 perf stat -e cycles,instructions \
 *       taskset -c 0-5 ./build/probe_v7 4096 6144 39 39
 */
#include <stdio.h>
#include <stdlib.h>

typedef float CONVFLOAT;
typedef int CONVINT;

void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output);

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s rows cols kernel_rows kernel_cols\n", argv[0]);
        return 2;
    }
    const CONVINT H = atoi(argv[1]);
    const CONVINT W = atoi(argv[2]);
    const CONVINT KH = atoi(argv[3]);
    const CONVINT KW = atoi(argv[4]);
    const CONVINT OH = H - KH + 1;
    const CONVINT OW = W - KW + 1;

    CONVFLOAT* in = (CONVFLOAT*)malloc((size_t)H * W * sizeof(CONVFLOAT));
    CONVFLOAT* k = (CONVFLOAT*)malloc((size_t)KH * KW * sizeof(CONVFLOAT));
    CONVFLOAT* out = (CONVFLOAT*)malloc((size_t)OH * OW * sizeof(CONVFLOAT));
    if (!in || !k || !out) return 1;

    for (size_t i = 0; i < (size_t)H * W; ++i) in[i] = (CONVFLOAT)((i % 23) * 0.031f - 0.2f);
    for (size_t i = 0; i < (size_t)KH * KW; ++i) k[i] = (CONVFLOAT)((i % 13) * 0.017f - 0.1f);

    conv2d(in, H, W, k, KH, KW, out);

    /* 阻止编译器把整段计算消掉 */
    volatile CONVFLOAT sink = out[0] + out[(size_t)OH * OW - 1];
    printf("checksum=%f\n", (double)sink);

    free(in); free(k); free(out);
    return 0;
}
