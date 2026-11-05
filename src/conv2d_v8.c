/* E13 VB=32：路数扫描中段
 *
 * 本文件由 conv2d_v7.c 只改 VB 常量得到，用于隔离「一次并行处理多少个输出」
 * 这个单一变量。除 VB 外，循环结构、寻址、尾块逻辑与 v7 完全一致。
 *
 * 用途：确认收益随 VB 单调上升的区间，不是 v7 的偶然值。
 *
 * 完整记录见 docs/experiments/E13-vector-width-sweep.md
 */

#include <string.h>
#include <math.h>
#include <omp.h>

typedef float CONVFLOAT;
typedef int CONVINT;

#define VB 32  /* 一次并行计算的输出个数 */

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

        CONVINT i = 0;
        for (; i + VB <= outputWidth; i += VB) {
            /* 8 个独立累加器：每个 lane 对应一个输出，互不归约 */
            CONVFLOAT sum[VB] = {0};

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* in_row = in_base + (size_t)jk * (size_t)inputWidth;
                const CONVFLOAT* k_row = kernel + (size_t)jk * (size_t)kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT kv = k_row[ik];
                    /* 内层对 t 展开：无跨 lane 依赖，可整条向量化 */
#pragma omp simd
                    for (int t = 0; t < VB; ++t) {
                        sum[t] += in_row[i + ik + t] * kv;
                    }
                }
            }
            for (int t = 0; t < VB; ++t) {
                out_row[i + t] = sum[t];
            }
        }
        /* 尾块：保持与 v1 完全相同的标量累加顺序 */
        for (; i < outputWidth; ++i) {
            CONVFLOAT s = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* in_row = in_base + (size_t)jk * (size_t)inputWidth;
                const CONVFLOAT* k_row = kernel + (size_t)jk * (size_t)kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    s += in_row[i + ik] * k_row[ik];
                }
            }
            out_row[i] = s;
        }
    }
}
