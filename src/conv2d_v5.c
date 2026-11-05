/* E09 二维输出分块：2 行 × 4 列
 *
 * 相对 v3（4 列 × 1 行）：
 *   一次同时累加 2 个相邻输出行的各 4 个输出，共 8 个累加器。
 *   核系数 kv 加载 1 次，被 8 个输出复用（v3 是 4 个）。
 *
 * 与 E07（8 路同行）的区别：
 *   E07 是「在同一行上加宽」，8 个输出共享同一批输入，输入 load 已经是
 *   一次向量载入，再加宽只会增加累加器压力；
 *   本实验是「换维度」，用两个输入行换更高的核系数复用率。
 *
 * 假设：
 *   当前瓶颈之一是每周期 load/广播比值；提高 kv 复用率可降低该比值。
 *
 * 证伪条件：
 *   1) GFLOPS 不高于 v3；2) 任一用例 FAIL 或误差非零。
 *
 * 结果：<待实测>
 */
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef float CONVFLOAT;
typedef int CONVINT;

/* 单行 4 路 */
static inline void row4(const CONVFLOAT* __restrict input,
                        const CONVFLOAT* __restrict kernel,
                        CONVINT inputWidth, CONVINT kernelHeight, CONVINT kernelWidth,
                        CONVINT j, CONVINT i, CONVFLOAT* __restrict out)
{
    CONVFLOAT s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
        const CONVFLOAT* inRow = input + (j + jk) * inputWidth + i;
        const CONVFLOAT* kRow = kernel + jk * kernelWidth;
        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
            const CONVFLOAT kv = kRow[ik];
            s0 += inRow[ik + 0] * kv;
            s1 += inRow[ik + 1] * kv;
            s2 += inRow[ik + 2] * kv;
            s3 += inRow[ik + 3] * kv;
        }
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

/* 单行单点 */
static inline CONVFLOAT row1(const CONVFLOAT* __restrict input,
                             const CONVFLOAT* __restrict kernel,
                             CONVINT inputWidth, CONVINT kernelHeight, CONVINT kernelWidth,
                             CONVINT j, CONVINT i)
{
    CONVFLOAT sum = 0.0f;
    for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
        const CONVFLOAT* inRow = input + (j + jk) * inputWidth + i;
        const CONVFLOAT* kRow = kernel + jk * kernelWidth;
        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
            sum += inRow[ik] * kRow[ik];
        }
    }
    return sum;
}

/* 2 行 × 4 列 */
static inline void block2x4(const CONVFLOAT* __restrict input,
                            const CONVFLOAT* __restrict kernel,
                            CONVINT inputWidth, CONVINT kernelHeight, CONVINT kernelWidth,
                            CONVINT j, CONVINT i,
                            CONVINT outWidth, CONVFLOAT* __restrict output)
{
    CONVFLOAT a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    CONVFLOAT b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
        const CONVFLOAT* r0 = input + (j + jk) * inputWidth + i;
        const CONVFLOAT* r1 = r0 + inputWidth;
        const CONVFLOAT* kRow = kernel + jk * kernelWidth;
        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
            const CONVFLOAT kv = kRow[ik];
            a0 += r0[ik + 0] * kv;
            a1 += r0[ik + 1] * kv;
            a2 += r0[ik + 2] * kv;
            a3 += r0[ik + 3] * kv;
            b0 += r1[ik + 0] * kv;
            b1 += r1[ik + 1] * kv;
            b2 += r1[ik + 2] * kv;
            b3 += r1[ik + 3] * kv;
        }
    }

    CONVFLOAT* o0 = output + j * outWidth + i;
    CONVFLOAT* o1 = o0 + outWidth;
    o0[0] = a0; o0[1] = a1; o0[2] = a2; o0[3] = a3;
    o1[0] = b0; o1[1] = b1; o1[2] = b2; o1[3] = b3;
}

void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output)
{
    const CONVINT OH = inputHeight - kernelHeight + 1;
    const CONVINT OW = inputWidth - kernelWidth + 1;

#pragma omp parallel for
    for (CONVINT j = 0; j < OH; ++j) {
        CONVINT i = 0;

        /* 两行一组，仅当还有下一行时走 2x4 */
        if (j + 1 < OH) {
            for (; i + 3 < OW; i += 4) {
                block2x4(input, kernel, inputWidth, kernelHeight, kernelWidth, j, i, OW, output);
            }
            for (; i < OW; ++i) {
                CONVFLOAT* o0 = output + j * OW + i;
                o0[0] = row1(input, kernel, inputWidth, kernelHeight, kernelWidth, j, i);
                o0[OW] = row1(input, kernel, inputWidth, kernelHeight, kernelWidth, j + 1, i);
            }
            ++j;  /* 本组两行已全部算完 */
        } else {
            for (; i + 3 < OW; i += 4) {
                row4(input, kernel, inputWidth, kernelHeight, kernelWidth, j, i,
                     output + j * OW + i);
            }
            for (; i < OW; ++i) {
                output[j * OW + i] = row1(input, kernel, inputWidth, kernelHeight, kernelWidth, j, i);
            }
        }
    }
}
