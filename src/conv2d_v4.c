/* E07 输出寄存器分块 8 路（相对 v3：主块 4 -> 8）
 * 结果：1024x768 k39 中位数 32.65 GFLOPS，比 4 路的 40.41 慢 19.2%，REJECT。
 * 无 spill（vmovups to stack = 0），瓶颈在 load/广播端口与调度压力。 */
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef float CONVFLOAT;
typedef int CONVINT;

void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVINT i = 0;

        for (; i + 7 < outputWidth; i += 8) {
            CONVFLOAT s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            CONVFLOAT s4 = 0, s5 = 0, s6 = 0, s7 = 0;

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* inRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT* kRow = kernel + jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT kv = kRow[ik];
                    s0 += inRow[ik + 0] * kv;
                    s1 += inRow[ik + 1] * kv;
                    s2 += inRow[ik + 2] * kv;
                    s3 += inRow[ik + 3] * kv;
                    s4 += inRow[ik + 4] * kv;
                    s5 += inRow[ik + 5] * kv;
                    s6 += inRow[ik + 6] * kv;
                    s7 += inRow[ik + 7] * kv;
                }
            }

            CONVFLOAT* out = output + j * outputWidth + i;
            out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
            out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
        }

        for (; i + 3 < outputWidth; i += 4) {
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
            CONVFLOAT* out = output + j * outputWidth + i;
            out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
        }

        for (; i < outputWidth; ++i) {
            CONVFLOAT sum = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* inRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT* kRow = kernel + jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    sum += inRow[ik] * kRow[ik];
                }
            }
            output[j * outputWidth + i] = sum;
        }
    }
}
