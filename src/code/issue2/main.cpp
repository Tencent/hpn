/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 * 
 * Author: moningchen@tencent.com
 * Content: Main Function For Issue 2
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>

// 检查是否支持TMA (需要Hopper架构 sm_90+)
#if __CUDA_ARCH__ >= 900 || !defined(__CUDA_ARCH__)
#include <cuda/pipeline>
#include <cooperative_groups.h>
#define TMA_SUPPORTED 1
#else
#define TMA_SUPPORTED 0
#endif

const int ITER_NUM = 100;

__global__ void lsu_copy_kernel(const float* src, float* dst, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void lsu_vectorized_copy_kernel(const float* src, float* dst, size_t n) {
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    float4* dst_vec = reinterpret_cast<float4*>(dst);
    
    size_t n_vec = n / 4;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += gridDim.x * blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
}

__global__ void lsu_vectorized_smem_copy_kernel(const float* src, float* dst, size_t n) {
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    float4* dst_vec = reinterpret_cast<float4*>(dst);
    
    extern __shared__ float4 smem_buffer[];
    
    size_t n_vec = n / 4;
    
    // 使用tile-based方式处理数据
    for (size_t tile_start = blockIdx.x * blockDim.x; tile_start < n_vec; tile_start += gridDim.x * blockDim.x) {
        size_t global_idx = tile_start + threadIdx.x;
        
        // Phase 1: 协作加载数据到shared memory
        if (global_idx < n_vec) {
            smem_buffer[threadIdx.x] = src_vec[global_idx];
        }
        
        // 同步确保所有数据加载完成
        __syncthreads();
        
        // Phase 2: 从shared memory写回到global memory
        if (global_idx < n_vec) {
            dst_vec[global_idx] = smem_buffer[threadIdx.x];
        }
        
        // 同步确保写操作完成后再进行下一轮
        __syncthreads();
    }
}

// TMA核函数 - 仅在Hopper+架构上可用
#if TMA_SUPPORTED && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ void tma_copy_kernel(const float* d_src, float* d_dst, size_t n) {
    extern __shared__ float smem[];
    float* s_buffer = smem;
    
    const size_t tile_size = blockDim.x;
    auto block = cooperative_groups::this_thread_block();
    
    // 创建barrier用于同步异步操作
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, block.size());
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    block.sync();
    
    // 处理每个tile
    for (size_t tile_idx = blockIdx.x; tile_idx * tile_size < n; tile_idx += gridDim.x) {
        size_t global_offset = tile_idx * tile_size;
        size_t elements_in_tile = min(tile_size, n - global_offset);
        
        if (elements_in_tile == 0) break;
        
        // 异步加载到共享内存
        if (threadIdx.x == 0) {
            cuda::memcpy_async(block, s_buffer, 
                             d_src + global_offset, 
                             elements_in_tile * sizeof(float),
                             barrier);
        }
        
        // // 等待异步复制完成
        barrier.arrive_and_wait();
        
        // 写回全局内存
        if (threadIdx.x < elements_in_tile) {
            d_dst[global_offset + threadIdx.x] = s_buffer[threadIdx.x];
        }
        block.sync();
    }
}
#else
__global__ void tma_copy_kernel(const float* d_src, float* d_dst, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        d_dst[i] = d_src[i];
    }
}
#endif

// 性能测试函数
double benchmark_lsu_scalar(const float* d_src, float* d_dst, size_t n, int iterations = 100) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 预热
    for (int i = 0; i < 3; i++) {
        lsu_copy_kernel<<<blocks_per_grid, threads_per_block>>>(d_src, d_dst, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        lsu_copy_kernel<<<blocks_per_grid, threads_per_block>>>(d_src, d_dst, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

double benchmark_lsu_vectorized(const float* d_src, float* d_dst, size_t n, int iterations = 100) {
    if (n % 4 != 0) return -1; // 不支持

    int threads_per_block = 256;
    int blocks_per_grid = ((n / 4) + threads_per_block - 1) / threads_per_block;

    // 预热
    for (int i = 0; i < 3; i++) {
        lsu_vectorized_copy_kernel<<<blocks_per_grid, threads_per_block>>>(d_src, d_dst, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        lsu_vectorized_copy_kernel<<<blocks_per_grid, threads_per_block>>>(d_src, d_dst, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

double benchmark_lsu_vectorized_smem(const float* d_src, float* d_dst, size_t n, int iterations = 100) {
    if (n % 4 != 0) return -1; // 不支持

    int threads_per_block = 256;
    int blocks_per_grid = ((n / 4) + threads_per_block - 1) / threads_per_block;
    
    // 计算动态shared memory大小
    size_t smem_size = threads_per_block * sizeof(float4);

    // 预热
    for (int i = 0; i < 3; i++) {
        lsu_vectorized_smem_copy_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(d_src, d_dst, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        lsu_vectorized_smem_copy_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(d_src, d_dst, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

double benchmark_ce_async(const float* d_src, float* d_dst, size_t n, int iterations = 100) {
    size_t bytes = n * sizeof(float);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 预热
    for (int i = 0; i < 3; i++) {
        cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    cudaEventRecord(stop);
    cudaStreamSynchronize(stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return milliseconds / iterations;
}

// TMA性能测试函数
double benchmark_tma(const float* d_src, float* d_dst, size_t n, int iterations = 100) {
#if TMA_SUPPORTED
    // 检查设备是否支持TMA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major < 9) {
        return -1; // 不支持
    }
    
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    size_t smem_size = threads_per_block * sizeof(float);

    // 预热
    for (int i = 0; i < 3; i++) {
        tma_copy_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(d_src, d_dst, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tma_copy_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(d_src, d_dst, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
#else
    return -1; // 不支持
#endif
}

// 性能评测函数
void run_performance_benchmark() {
    printf("\n=== GPU内通信方式性能评测 ===\n");
    
    struct {
        size_t bytes;
        const char* label;
    } message_sizes[] = {
        {4 * 1024, "4KB"},
        {64 * 1024, "64KB"},
        {1024 * 1024, "1MB"},
        {16 * 1024 * 1024, "16MB"},
        {64 * 1024 * 1024, "64MB"},
        {256 * 1024 * 1024, "256MB"},
        {512 * 1024 * 1024, "512MB"},
        {1024ULL * 1024 * 1024, "1GB"},
        {2048ULL * 1024 * 1024, "2GB"},
        {4096ULL * 1024 * 1024, "4GB"}
    };
    int num_sizes = sizeof(message_sizes) / sizeof(message_sizes[0]);

    printf("Message Size\tLSU Scalar\tLSU Vector\tLSU Vec+SMem\tCE Async\tTMA\t\tBandwidth(GB/s)\n");
    printf("           \t(ms)     \t(ms)     \t(ms)       \t(ms)    \t(ms)\t\t(Best)     \n");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t bytes = message_sizes[i].bytes;
        size_t n = bytes / sizeof(float);

        // 检查GPU内存是否足够
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (bytes * 2 > free_mem) {
            printf("%s\t\tSkipped (Insufficient GPU memory: need %zuMB, available %zuMB)\n", 
                   message_sizes[i].label, (bytes * 2) / (1024 * 1024), free_mem / (1024 * 1024));
            continue;
        }

        printf("Testing %s...\n", message_sizes[i].label);

        // 分配设备内存
        float *d_src, *d_dst;
        cudaError_t err1 = cudaMalloc(&d_src, bytes);
        cudaError_t err2 = cudaMalloc(&d_dst, bytes);
        
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            printf("%s\t\tSkipped (GPU memory allocation failed)\n", message_sizes[i].label);
            if (err1 == cudaSuccess) cudaFree(d_src);
            if (err2 == cudaSuccess) cudaFree(d_dst);
            continue;
        }

        // 初始化数据（只初始化一小部分，然后复制填充）
        size_t init_size = min(n, (size_t)(1024 * 1024)); // 最多初始化1M个元素
        float *h_init = (float*)malloc(init_size * sizeof(float));
        for (size_t j = 0; j < init_size; j++) {
            h_init[j] = (float)j;
        }
        
        // 将初始数据复制到GPU，然后在GPU内部复制填充整个数组
        cudaMemcpy(d_src, h_init, init_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 使用CE填充剩余数据
        if (n > init_size) {
            for (size_t offset = init_size; offset < n; offset += init_size) {
                size_t copy_size = min(init_size, n - offset);
                cudaMemcpy(d_src + offset, d_src, copy_size * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }

        int iterations = ITER_NUM;

        // 运行基准测试
        double lsu_scalar_time = benchmark_lsu_scalar(d_src, d_dst, n, iterations);
        double lsu_vector_time = (n % 4 == 0) ? benchmark_lsu_vectorized(d_src, d_dst, n, iterations) : -1;
        double lsu_vector_smem_time = (n % 4 == 0) ? benchmark_lsu_vectorized_smem(d_src, d_dst, n, iterations) : -1;
        double ce_async_time = benchmark_ce_async(d_src, d_dst, n, iterations);
        double tma_time = benchmark_tma(d_src, d_dst, n, iterations);

        // 找到最佳时间
        double best_time = lsu_scalar_time;
        if (lsu_vector_time > 0 && lsu_vector_time < best_time) best_time = lsu_vector_time;
        if (lsu_vector_smem_time > 0 && lsu_vector_smem_time < best_time) best_time = lsu_vector_smem_time;
        if (ce_async_time < best_time) best_time = ce_async_time;
        if (tma_time > 0 && tma_time < best_time) best_time = tma_time;
        
        // 计算带宽 (GB/s) - 包括读和写操作
        double bandwidth = (bytes * 2.0) / (best_time * 1e-3) / 1e9;

        printf("%s\t\t%.3f\t\t", message_sizes[i].label, lsu_scalar_time);
        if (lsu_vector_time > 0) {
            printf("%.3f\t\t", lsu_vector_time);
        } else {
            printf("N/A\t\t");
        }
        if (lsu_vector_smem_time > 0) {
            printf("%.3f\t\t", lsu_vector_smem_time);
        } else {
            printf("N/A\t\t");
        }
        printf("%.3f\t\t", ce_async_time);
        if (tma_time > 0) {
            printf("%.3f\t\t", tma_time);
        } else {
            printf("N/A\t\t");
        }
        printf("%.2f\n", bandwidth);

        // 清理
        free(h_init);
        cudaFree(d_src);
        cudaFree(d_dst);
    }
}

int main() {
    // 设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
#if TMA_SUPPORTED
    if (prop.major >= 9) {
        printf("TMA Support: YES (Hopper architecture detected)\n");
    } else {
        printf("TMA Support: NO (Requires Hopper architecture sm_90+)\n");
    }
#else
    printf("TMA Support: NO (Not compiled with sm_90+ target)\n");
    printf("To enable TMA, compile with: nvcc -arch=sm_90 or higher\n");
#endif
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f GB total, %.2f GB available\n", 
           total_mem / 1e9, free_mem / 1e9);

    // 运行性能基准测试
    run_performance_benchmark();

    return 0;
}