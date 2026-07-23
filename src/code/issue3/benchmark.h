/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: CUDA Event 计时器 — 基于 NCCL 模式的 GPU kernel 计时
 ************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

struct CudaTimer {
    cudaEvent_t start_ev, stop_ev;

    CudaTimer() {
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_ev, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_ev, stream);
    }

    float elapsed() {
        float ms;
        cudaEventSynchronize(stop_ev);
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        return ms;
    }
};
