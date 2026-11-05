/* E02 行基址指针替代二维下标（对照实验，预期被证伪）
 *
 * 相对 v1：
 *   唯一改动是把 (j+jk)*inputWidth + (i+ik) 这类二维下标，
 *   替换为「每层循环外提一次行基址，内层用行内偏移」的写法。
 *   累加器位置、循环顺序、float 累加链、写回时机全部与 v1 一致。
 *
 * 假设：
 *   二维下标里的整数乘法与地址重算是可观开销，显式提行基址能降低它。
 *
 * 证伪条件（任一成立即假设不成立）：
 *   1) 四案 GFLOPS 变化落在噪声带内（< 2 倍抖动）；
 *   2) 任一变体变慢；
 *   3) 任用例正确性 FAIL 或误差非零。
 *
 * 设计意图：
 *   这是一个「预期失败」的对照实验。现代编译器在 -O3 下早已完成
 *   强度削减与地址公共子表达式消除，手写行指针多数时候是零收益甚至负收益。
 *   用它校准判断力：凭直觉猜优化点是常见的错误路径。
 *
 * 结果：<待实测>
 * 代价：几乎为零；若失败，说明该类改动不值得占用优化预算。
 *
 * 完整记录见 docs/experiments/E02-row-pointer.md
 */

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
        const CONVFLOAT* in_base = input + (size_t)j * (size_t)inputWidth;
        CONVFLOAT* out_row = output + (size_t)j * (size_t)outputWidth;
        for (CONVINT i = 0; i < outputWidth; ++i) {
            CONVFLOAT sum = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* in_row = in_base + (size_t)jk * (size_t)inputWidth;
                const CONVFLOAT* k_row = kernel + (size_t)jk * (size_t)kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    sum += in_row[i + ik] * k_row[ik];
                }
            }
            out_row[i] = sum;
        }
    }
}
