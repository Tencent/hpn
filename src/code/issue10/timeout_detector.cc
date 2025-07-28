#include "include/timeout_detector.h"

std::map<ncclComm_t, bool> NCCLKernelTimeoutDetector::activeComms;
std::mutex NCCLKernelTimeoutDetector::globalMutex;
std::atomic<bool> NCCLKernelTimeoutDetector::detectorRunning{false};
std::thread NCCLKernelTimeoutDetector::detectorThread;