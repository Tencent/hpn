

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
