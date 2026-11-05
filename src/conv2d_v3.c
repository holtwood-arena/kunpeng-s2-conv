/* E04 输出寄存器分块：沿 i 方向一次累加 4 个输出
 *
 * 相对 v1：
 *   内层每次取一个核系数 kv，同时更新 4 个相邻输出的累加器。
 *   核加载由 4 次降为 1 次，地址计算由 4 次降为 1 次基址 + 常数偏移，
 *   并形成 4 条独立累加链以提高 ILP。
 *
 * 假设：
 *   当前实现每个输出独立走完整 (jk,ik) 双循环，核系数被重复加载、
 *   且单条依赖链限制了指令级并行。
 *
 * 证伪条件：
 *   1) GFLOPS 提升落在噪声带内；
 *   2) 任一用例正确性 FAIL 或误差非零。
 *
 * 说明：
 *   本实现采用行基址指针，这是分块的自然写法。
 *   E02「单独把二维下标换成行指针」的结论（无收益）仍然成立，
 *   二者不冲突：收益来自分块与复用，不是指针本身。
 *
 * 结果：四案 PASS，误差 0.00e+00；中位数 44.31 / 45.07 / 46.11 / 45.80 GFLOPS
 *       相对 v0 基线 ×3.26~3.37（LOCAL_REFERENCE, Ryzen 7 5800H 6 线程）
 */
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

        /* 主块：一次 4 个输出 */
        for (; i + 3 < outputWidth; i += 4) {
            CONVFLOAT s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

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
            out[0] = s0;
            out[1] = s1;
            out[2] = s2;
            out[3] = s3;
        }

        /* 尾块 */
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
