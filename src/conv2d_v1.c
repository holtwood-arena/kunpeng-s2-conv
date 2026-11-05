/* E01 局部累加器 + 删除冗余 memset
 *
 * 相对 v0：
 *   内层 output[...] += input*kernel 改为局部 float sum 累加，循环结束写回一次；
 *   删除冗余 memset（局部累加后输出无需初值）。
 *
 * 假设：
 *   原内层是 read-modify-write，且 input/kernel/output 为裸指针缺少别名信息，
 *   编译器无法把输出提升到寄存器。
 *
 * 证伪条件（任一成立即假设不成立）：
 *   1) 反汇编内层仍存在对输出地址的访存；
 *   2) 四案 GFLOPS 变化全部落在噪声带内；
 *   3) 任用例正确性 FAIL；
 *   4) 最大绝对误差从 0.00e+00 变为非零。
 *
 * 结果：<待实测>
 * 代价：删除 memset 依赖「每个输出必被覆盖一次」，需 OH>0 && OW>0（赛题保证）。
 *
 * 完整记录见 docs/experiments/E01-local-accumulator.md
 */

#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

// 核心实现：局部累加器（E01）
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth, CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        for (CONVINT i = 0; i < outputWidth; ++i) {
            // E01: 累加器留在寄存器，避免每个 MAC 都读写输出。
            // 严格保持 float 累加，与 v0 的累加顺序和精度一致。
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
