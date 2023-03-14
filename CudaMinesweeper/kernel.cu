
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <cstdio>
#include <exception>
#include <algorithm>
#include <memory.h>
#include <Windows.h>

constexpr int W = 9;
constexpr int H = 9;
constexpr int MineCount = 10;
constexpr int ClickX = 0;
constexpr int ClickY = 0;

/// <summary>
/// 扫雷棋盘的模板信息
/// </summary>
struct TemplateInfo {
    // 每个元素表示当前单元格附近有多少个单元格。（只可能是3,5,8）
    uint8_t counts[W * H];
    // 每个元素表示当前单元格附近所有单元格。
    uint8_t offsets[W * H][8];
    // 除点击单元格的其他所有单元格。例如初始点击(1,0)，那么 reindex = [0,2,3,4,5,6,...]
    uint8_t reindex[W * H - 1];

    __host__ static TemplateInfo create() {
        TemplateInfo ret;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                uint8_t* list = ret.offsets[y * W + x];
                uint8_t count = 0;
                for (int yy = y - 1; yy <= y + 1; yy++) {
                    if (yy < 0 || yy >= H) continue;
                    for (int xx = x - 1; xx <= x + 1; xx++) {
                        if (yy == y && xx == x) continue;
                        if (xx < 0 || xx >= W) continue;
                        list[count++] = yy * W + xx;
                    }
                }
                ret.counts[y * W + x] = count;
            }
        }

        uint8_t index = 0;
        for (int i = 0; i < W * H - 1; i++, index++) {
            if (i == ClickY * W + ClickX) index++;
            ret.reindex[i] = index;
        }

        return ret;
    }
};

/// <summary>
/// 求当前组合序列的下一个组合序列
/// </summary>
/// <param name="indices"></param>
/// <returns></returns>
__device__ __host__ static void nextIndices(uint8_t indices[MineCount]) {
    for (int i = MineCount - 1, m = W * H - 1; ; i--, m--) {
        if (++indices[i] < m) {
            int begin = indices[i++];
            for (; i < MineCount; i++) {
                indices[i] = ++begin;
            }
            break;
        }
    }
}

struct Minesweeper {
    int8_t map[W * H];
    uint8_t stack[W * H];
    uint8_t indices[MineCount];


    __device__ Minesweeper(const uint8_t indices[MineCount]) {
        memcpy(this->indices, indices, sizeof this->indices);
    }

    __device__ void init(const TemplateInfo& info) {
        memset(map, 0, sizeof map);

        for (int i = 0; i < MineCount; i++) {
            int mineIndex = info.reindex[indices[i]];
            map[mineIndex] = -128;
            int offsetCount = info.counts[mineIndex];
            const uint8_t* offsets = info.offsets[mineIndex];
            for (int k = 0; k < offsetCount; k++) {
                map[offsets[k]]++;
            }
        }
    }

    __device__ bool click(const TemplateInfo& info) {
        constexpr int8_t ClickMask = 0x40;

        stack[0] = ClickY * W + ClickX;
        uint8_t stackPointer = 1;
        map[ClickY * W + ClickX] |= ClickMask;
        int clickCount = 0;

        while (stackPointer) {
            int currentOffset = stack[--stackPointer];
            clickCount++;
            if (map[currentOffset] & ~ClickMask) continue;

            int offsetCount = info.counts[currentOffset];
            const uint8_t* offsets = info.offsets[currentOffset];
            for (int i = 0; i < offsetCount; i++) {
                if (map[offsets[i]] & ClickMask) continue;
                map[offsets[i]] |= ClickMask;
                stack[stackPointer++] = offsets[i];
            }
        }

        return clickCount == W * H - MineCount;
    }

    __device__ __host__ void print() {
        char buffer[(W + 1) * H + 2];

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (x == ClickX && y == ClickY) {
                    buffer[y * (W + 1) + x] = '~';
                    continue;
                }

                int8_t v = map[y * W + x] & ~0x40;
                if (v < 0) {
                    buffer[y * (W + 1) + x] = '*';
                } else if (v > 0) {
                    buffer[y * (W + 1) + x] = (v & 15) + '0';
                } else {
                    buffer[y * (W + 1) + x] = '.';
                }
            }
            buffer[y * (W + 1) + W] = '\n';
        }
        buffer[(W + 1) * H] = '\n';
        buffer[(W + 1) * H + 1] = '\0';

        printf("%s", buffer);
    }
};

/// <summary>
/// 二项式系数，用于加速组合序列的计算。
/// </summary>
struct BinomialCoefficient {
    uint64_t values[MineCount][W * H];

    __host__ static BinomialCoefficient create() {
        BinomialCoefficient ret;

        for (int i = 0; i < W * H; i++) ret.values[0][i] = 1;

        for (int r = 1; r < MineCount; r++) {
            ret.values[r][r - 1] = 0;
            for (int c = r; c < W * H; c++) {
                ret.values[r][c] = ret.values[r - 1][c - 1] + ret.values[r][c - 1];
            }
        }

        return ret;
    }
};

/// <summary>
/// 根据组合序号(order)得到组合序列。
/// </summary>
/// <param name="bc">二项式系数，用于加速计算。</param>
/// <param name="indices">组合序列</param>
/// <param name="order">组合序号</param>
/// <returns></returns>
__device__ __host__ static void sortedCombination(const BinomialCoefficient& bc, uint8_t indices[MineCount], uint64_t order) {
    uint8_t index = 0;
    for (int i = 0, n = MineCount - 1; i < MineCount; i++, n--) {
        for (int m = W * H - 2 - index; m >= n && order >= bc.values[n][m]; m--, index++) {
            order -= bc.values[n][m];
        }
        indices[i] = index++;
    }
}

__device__ static BinomialCoefficient binomialCoefficient;
__device__ static TemplateInfo templateInfo;

__global__ static void initIndicesKernel(uint8_t(*indices)[MineCount], uint64_t order, uint64_t step) {
    __shared__ BinomialCoefficient bc;
    if (threadIdx.x == 0) {
        bc = binomialCoefficient;
    }

    __syncthreads();

    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    sortedCombination(bc, indices[i], order + i * step);
}

__global__ static void enumerateKernel(uint8_t(*indices)[MineCount], uint64_t steps, uint64_t* result, bool useNext) {
    __shared__ TemplateInfo templateInfo;
    if (threadIdx.x == 0) {
        templateInfo = ::templateInfo;
    }

    __syncthreads();

    size_t batchIndex = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    Minesweeper mine(indices[batchIndex]);
    if (useNext) nextIndices(mine.indices);
    mine.init(templateInfo);
    if (mine.click(templateInfo)) atomicAdd(result, 1);

    for (uint64_t i = 1; i < steps; i++) {
        nextIndices(mine.indices);
        mine.init(templateInfo);
        if (mine.click(templateInfo)) atomicAdd(result, 1);
    }

    memcpy(indices[batchIndex], mine.indices, sizeof indices[batchIndex]);
}

template<typename = void>
struct ErrorHandler {
    struct exception : public std::exception {
        using std::exception::exception;
    };

    ErrorHandler(cudaError_t err) {
        if (err != cudaSuccess) {
            const char* msg = cudaGetErrorString(err);
            fprintf(stderr, "%s", msg);
            throw exception(msg, static_cast<int>(err));
        }
    }
};

struct Enumerator {
    uint64_t order;
    uint64_t stepIndex = 0;
    uint64_t stepCount;
    size_t blockCount;
    size_t threadPerBlockCount;
    uint8_t(*devIndices)[MineCount] = nullptr;
    uint64_t* devResultCount = nullptr;

    Enumerator(uint64_t order, uint64_t stepCount, size_t blockCount, size_t threadPerBlockCount) :
        order(order),
        stepCount(stepCount),
        blockCount(blockCount),
        threadPerBlockCount(threadPerBlockCount)
    {
        try {
            (ErrorHandler<>)cudaMalloc(&devIndices, MineCount * sizeof(uint8_t) * blockCount * threadPerBlockCount);
            (ErrorHandler<>)cudaMalloc(&devResultCount, sizeof(uint64_t));

            uint64_t resultCount = 0;
            (ErrorHandler<>)cudaMemcpy(devResultCount, &resultCount, sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
        catch (std::exception) {
            this->~Enumerator();
            throw;
        }
    }

    ~Enumerator() {
        cudaFree(devIndices);
        cudaFree(devResultCount);
    }

    void next(uint64_t steps = 1) {
        steps = std::min(steps, stepCount - stepIndex);
        if (steps == 0) return;

        if (stepIndex == 0) {
            initIndicesKernel<<<blockCount, threadPerBlockCount>>>(devIndices, order, stepCount);
        }
        enumerateKernel<<<blockCount, threadPerBlockCount>>>(devIndices, steps, devResultCount, stepIndex != 0);
        stepIndex += steps;
    }

    uint64_t resultCount() const {
        uint64_t result = 0;
        (ErrorHandler<>)cudaMemcpy(&result, devResultCount, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return result;
    }
};

static uint64_t enumerate(uint64_t order, uint64_t count, uint64_t* endOrder) {
    size_t threadPerBlockCount = 256;
    size_t blockCount = 1ULL * 1024 * 1024 * 1024 / (threadPerBlockCount * MineCount);
    size_t threadCount = threadPerBlockCount * blockCount;
    uint64_t stepCount = count / threadCount;

    if (stepCount == 0) {
        blockCount = count / (threadPerBlockCount * 100);
        if (blockCount != 0) {
            threadCount = threadPerBlockCount * blockCount;
            stepCount = count / threadCount;
        } else {
            threadPerBlockCount = 1;
            blockCount = count;
            threadCount = count;
            stepCount = 1;
        }
    }
    if (endOrder) *endOrder = order + stepCount * threadCount;

    Enumerator enumer(order, stepCount, blockCount, threadPerBlockCount);
    LARGE_INTEGER t1, t2, freq;
    QueryPerformanceFrequency(&freq);

    while (enumer.stepIndex < enumer.stepCount) {
        QueryPerformanceCounter(&t1);
        enumer.next(100);
        uint64_t count = enumer.resultCount();
        QueryPerformanceCounter(&t2);
        printf("%lld/%lld, result = %lld, %.3fs\n", enumer.stepIndex, enumer.stepCount, count, (t2.QuadPart - t1.QuadPart) / (double)freq.QuadPart);
    }
    return enumer.resultCount();
}

// 9x9 (0,0) : 18897794 / 1646492110120 ≈ 0.000011478
// 9x9 (4,4) :  9672404 / 1646492110120 ≈ 0.000005875
// 8x8 (0,0) :   429889 /  127805525001 ≈ 0.000003364
// 8x8 (4,4) :   269866 /  127805525001 ≈ 0.000002112
// 7x7 (0,0) :    10549 /    6540715896

int main() {
    TemplateInfo templateInfo = TemplateInfo::create();
    (ErrorHandler<>)cudaMemcpyToSymbol(::templateInfo, &templateInfo, sizeof templateInfo);
    BinomialCoefficient binomialCoefficient = BinomialCoefficient::create();
    (ErrorHandler<>)cudaMemcpyToSymbol(::binomialCoefficient, &binomialCoefficient, sizeof binomialCoefficient);

    uint64_t total = 1;
    for (int i = 1; i <= MineCount; i++) {
        total = total * (W * H - i) / i;
    }
    printf("total: %lld\n", total);

    uint64_t result = 0;
    uint64_t order = 0;
    while (order != total) {
        result += enumerate(order, total - order, &order);
    }
    printf("result: %lld\n", result);
}
