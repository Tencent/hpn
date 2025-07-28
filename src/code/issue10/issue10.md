NCCL Kernel 超时异常检测机制

本项目完成了 NCCL 通信中的一个可观测性问题 —— 在 kernel 卡住但 host 无感知的情况下，自动检测并输出异常日志。这是多机多卡训练环境中常见且难以定位的问题。当前工作专注于kernel 卡住不报错，并通过引入事件记录机制和后台监测线程，在不影响性能的前提下，实现了 NCCL kernel 的运行状态监控。

思路：在 NCCL 启动每个 kernel 时记录启动事件，同时在相应 stream 上注册 cuda event，kernel 运行完后记录结束事件；由后台线程定期轮询是否有 kernel 长时间未完成。

如果发现某个通信器启动后超过阈值（默认 2000ms）仍未完成，并且 start event 已触发而 end event 尚未触发，就认为它已超时。此时会输出详细的诊断日志，包括通信器 rank、核名、线程配置、channel mask 等关键信息，并主动设置 async error 标记，便于上层感知错误。

整个检测逻辑通过一个独立线程实现，不插手 NCCL 通信流程本身，因此不会对正常性能造成干扰。为了验证效果，我们还设计了 ncclSimulateKernelTimeout 函数，在实际运行中主动挂起一个“假 kernel”，用于测试检测机制是否生效。

关键点
​    •   使用 cudaEventRecord + cudaEventQuery 检查 kernel 执行状态

​    •   引入 comm->kernelTracker 结构保存每个 kernel 的关键元信息（启动时间、线程数等）

​    •   多线程安全访问通过 std::mutex 和 std::atomic 管理

​    •   检测线程独立于主流程运行，确保低侵入性

​    •   所有相关资源在 ncclCommDestroy 中做了完整清理，避免泄漏或误报

由于是host侧的代码插桩，因此device侧的GPU性能不受影响，而CPU仅仅新增了一个每秒一次的轮询，影响也非常小。

测试命令：

```
root@autodl-container-93b9118200-e75b2e49:~/nccl-debug/nccl-tests# ./build/all_reduce_perf -b 8G -e 8G -f 1 -g 2 -n 2
```



补充，下面是新增的代码：

debug.cc:

```
...
#include "timeout_detector.h"
...
ncclResult_t ncclSimulateKernelTimeout(ncclComm_t comm) {

    cudaStream_t testStream;
    CUDACHECK(cudaStreamCreate(&testStream));
    
    NCCLCHECK(NCCLKernelTimeoutDetector::recordKernelStart(comm, "TestTimeoutKernel", 0x1, 256));
    
    // 不启动真正的kernel，让检测器超时
    
    std::this_thread::sleep_for(std::chrono::milliseconds(
        comm->kernelTracker.timeoutThresholdMs + 2000));
    
    CUDACHECK(cudaStreamDestroy(testStream));
    return ncclSuccess;
}
```

enqueue.cc

```
...
#include "timeout_detector.h"
...
const char* kernelName = "sqq";
ncclSimulateKernelTimeout(comm);
NCCLCHECKGOTO(NCCLKernelTimeoutDetector::recordKernelStart(
   comm, kernelName, plan->channelMask, plan->threadPerBlock), ret, do_return);
...

NCCLCHECK(NCCLKernelTimeoutDetector::recordKernelEnd(comm, launchStream));
...
```

init.cc:

```
#include <mutex>
#include "debug.h"
#include "comm.h"
#include "timeout_detector.h"
...
NCCLKernelTimeoutDetector::initTimeoutDetector(comm);
...
ncclCommTimeoutCleanup(comm);
...
ncclResult_t ncclCommTimeoutCleanup(ncclComm_t comm) {

    std::lock_guard<std::mutex> lock(NCCLKernelTimeoutDetector::globalMutex);

    NCCLKernelTimeoutDetector::activeComms.erase(comm);

    if (comm->kernelTracker.startEvent) {
        CUDACHECK(cudaEventDestroy(comm->kernelTracker.startEvent));
    }
    if (comm->kernelTracker.endEvent) {
        CUDACHECK(cudaEventDestroy(comm->kernelTracker.endEvent));
    }

    return ncclSuccess;
}
...
```

Makefile:

```
...
LIBSRCFILES := \
	bootstrap.cc channel.cc collectives.cc debug.cc enqueue.cc group.cc \
	init.cc init_nvtx.cc proxy.cc transport.cc mnnvl.cc allocator.cc symmetric.cc timeout_detector.cc \
...
```

timeout_detector.cc:

```
#include "include/timeout_detector.h"

std::map<ncclComm_t, bool> NCCLKernelTimeoutDetector::activeComms;
std::mutex NCCLKernelTimeoutDetector::globalMutex;
std::atomic<bool> NCCLKernelTimeoutDetector::detectorRunning{false};
std::thread NCCLKernelTimeoutDetector::detectorThread;
```

comm.h:

```
#include <chrono>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>
...
struct ncclComm;
typedef struct ncclComm* ncclComm_t;

ncclResult_t ncclCommTimeoutCleanup(ncclComm_t comm);
...
#define NCCL_KERNEL_TIMEOUT_DEFAULT_MS 2000  // 2秒
#define NCCL_KERNEL_TIMEOUT_CHECK_INTERVAL_MS 1000  // 1秒检查间隔

struct ncclKernelTimeoutTracker {
  cudaEvent_t startEvent;
  cudaEvent_t endEvent;
  std::chrono::steady_clock::time_point launchTime;
  bool kernelActive;
  std::atomic<bool> timeoutDetected;
  int timeoutThresholdMs;
  const char* kernelName;
  int channelMask;
  int threadPerBlock;
};

static inline int ncclGetTimeoutThreshold() {
    const char* timeoutStr = getenv("NCCL_KERNEL_TIMEOUT_MS");
    if (timeoutStr) {
        int timeout = atoi(timeoutStr);
        if (timeout > 0) return timeout;
    }
    return NCCL_KERNEL_TIMEOUT_DEFAULT_MS;
}
...
struct ncclKernelTimeoutTracker kernelTracker;
  std::mutex timeoutMutex;
...
```

debug.h:

```
...
ncclResult_t ncclSimulateKernelTimeout(ncclComm_t comm);
...
```

timeout_detector.h:

```

#include "comm.h"
#include "debug.h"
#include "checks.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>

class NCCLKernelTimeoutDetector {
  public:
      static std::map<ncclComm_t, bool> activeComms;
      static std::mutex globalMutex;
      static std::atomic<bool> detectorRunning;
      static std::thread detectorThread;

  public:
      static ncclResult_t initTimeoutDetector(ncclComm_t comm) {

          std::lock_guard<std::mutex> lock(globalMutex);
          
          CUDACHECK(cudaEventCreateWithFlags(&comm->kernelTracker.startEvent, cudaEventDisableTiming));
          CUDACHECK(cudaEventCreateWithFlags(&comm->kernelTracker.endEvent, cudaEventDisableTiming));

          comm->kernelTracker.kernelActive = false;
          comm->kernelTracker.timeoutDetected.store(false);
          comm->kernelTracker.timeoutThresholdMs = ncclGetTimeoutThreshold();
          
          activeComms[comm] = true;
          
          if (!detectorRunning.load()) {
              detectorRunning.store(true);
              detectorThread = std::thread(timeoutDetectorLoop);
          }
          
          return ncclSuccess;
      }
      
      static ncclResult_t recordKernelStart(ncclComm_t comm, const char* kernelName, 
                                          int channelMask, int threadPerBlock) {

                  
          std::lock_guard<std::mutex> lock(comm->timeoutMutex);
          
          comm->kernelTracker.launchTime = std::chrono::steady_clock::now();
          comm->kernelTracker.kernelActive = true;
          comm->kernelTracker.timeoutDetected.store(false);
          comm->kernelTracker.kernelName = kernelName;
          comm->kernelTracker.channelMask = channelMask;
          comm->kernelTracker.threadPerBlock = threadPerBlock;
          
          cudaStream_t currentStream = comm->planner.streams->stream;
          CUDACHECK(cudaEventRecord(comm->kernelTracker.startEvent, currentStream));
          
          return ncclSuccess;
      }
      
      static ncclResult_t recordKernelEnd(ncclComm_t comm, cudaStream_t stream) {
          std::lock_guard<std::mutex> lock(comm->timeoutMutex);
          
          if (comm->kernelTracker.kernelActive) {
              CUDACHECK(cudaEventRecord(comm->kernelTracker.endEvent, stream));
              comm->kernelTracker.kernelActive = false;
          }
          
          return ncclSuccess;
      }
      
      static bool checkCommTimeout(ncclComm_t comm) {
          std::lock_guard<std::mutex> lock(comm->timeoutMutex);
          
          if (!comm->kernelTracker.kernelActive) {
              return false;
          }
          
          auto now = std::chrono::steady_clock::now();
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              now - comm->kernelTracker.launchTime).count();
              
          if (elapsed > comm->kernelTracker.timeoutThresholdMs) {
              cudaError_t startStatus = cudaEventQuery(comm->kernelTracker.startEvent);
              cudaError_t endStatus = cudaEventQuery(comm->kernelTracker.endEvent);
              
              if (startStatus == cudaSuccess && endStatus == cudaErrorNotReady) {
                  logKernelTimeout(comm, elapsed);
                  comm->kernelTracker.timeoutDetected.store(true);
                  return true;
              }
          }
          
          return false;
      }
      
      static void logKernelTimeout(ncclComm_t comm, long elapsedMs) {
          INFO(NCCL_ALL, "NCCL KERNEL TIMEOUT DETECTED");
          INFO(NCCL_ALL, "=== NCCL Kernel Timeout Report ===");
          INFO(NCCL_ALL, "Comm: %p, Rank: %d/%d", comm, comm->rank, comm->nRanks);
          INFO(NCCL_ALL, "Kernel: %s", comm->kernelTracker.kernelName ? comm->kernelTracker.kernelName : "Unknown");
          INFO(NCCL_ALL, "Timeout: %ld ms (threshold: %d ms)", elapsedMs, comm->kernelTracker.timeoutThresholdMs);
          INFO(NCCL_ALL, "Channel Mask: 0x%x", comm->kernelTracker.channelMask);
          INFO(NCCL_ALL, "Threads per Block: %d", comm->kernelTracker.threadPerBlock);
          INFO(NCCL_ALL, "CUDA Device: %d", comm->cudaDev);
          INFO(NCCL_ALL, "==================================");
          
          (void)ncclCommSetAsyncError(comm, ncclSystemError);
      }
      
  private:
      static void timeoutDetectorLoop() {
          while (detectorRunning.load()) {
              std::this_thread::sleep_for(std::chrono::milliseconds(NCCL_KERNEL_TIMEOUT_CHECK_INTERVAL_MS));
              
              std::lock_guard<std::mutex> lock(globalMutex);
              for (auto& pair : activeComms) {
                  if (pair.second) {
                      checkCommTimeout(pair.first);
                  }
              }
          }
      }
      
};
```

