/* E12 输出维度向量化（8 输出 × SIMD lane）
 *
 * 背景（来自 v3 的汇编证据）：
 *   v3 的 4 路分块被 GCC 向量化到了 **ik（归约）维度**：
 *   主循环是 4 条 vmulps(ymm) 算 4 个重叠窗口，随后用一大串
 *   vunpcklps / vshufps / vextractf128 做交错合并，最后才标量累加。
 *   perf 实测：IPC 2.34，但 FLOP/cycle 只有约 1.9 —— 大部分指令花在 shuffle 上。
 *
 * 假设：
 *   把向量化维度从「归约维度 ik」换成「输出维度 i」，
 *   让 8 个不同输出共享同一条向量累加器，就不需要任何交错/水平归约。
 *   而且每个 lane 对应一个独立输出，其 jk→ik 累加顺序与标量版本逐位一致，
 *   所以精度应当仍是 0 误差。
 *
 * 证伪条件：
 *   1) GFLOPS 相对 v3 提升小于 2 倍抖动；
 *   2) 任一用例 FAIL 或误差非零。
 *
 * 结果：<待实测>
 * 代价：绑定向量宽度（8 路需 256-bit；128-bit NEON 上只能 4 路）。
 */

#include <string.h>
#include <math.h>
#include <omp.h>

typedef float CONVFLOAT;
typedef int CONVINT;

#define VB 8  /* 一次并行计算的输出个数 */

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
