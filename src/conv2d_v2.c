/* E05 解锁编译器自动向量化：__restrict 消除别名假设
 *
 * 相对 v1：
 *   只在三个指针参数上加 __restrict，其余逻辑一字未改。
 *
 * 假设：
 *   v0/v1 未被自动向量化的主因，是编译器无法证明 input/kernel/output 不重叠，
 *   因而不能重排跨迭代的访存。
 *
 * 证伪条件：
 *   1) -fopt-info-vec 仍报告 not vectorized；
 *   2) GFLOPS 变化落在噪声带内；
 *   3) 任一用例正确性 FAIL。
 *
 * 前提：调用方保证三块内存不重叠（bench_conv.c 用独立 malloc，成立）。
 * 结果：编译器报告 vectorized，但实测 -3.0%（15.19 -> 14.73），REJECT
 */
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef float CONVFLOAT;
typedef int CONVINT;

void conv2d(const CONVFLOAT* __restrict input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* __restrict kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* __restrict output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        for (CONVINT i = 0; i < outputWidth; ++i) {
            CONVFLOAT sum = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONVINT inputIdx = (j + jk) * inputWidth + (i + ik);
                    CONVINT kernelIdx = jk * kernelWidth + ik;
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
            output[j * outputWidth + i] = sum;
        }
    }
}
