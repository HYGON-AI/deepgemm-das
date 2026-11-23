// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// Persistent spinning kernel: occupies up to <nblocks> blocks on a device
// until killed. Usage: spin_occupier [dev] [nblocks]
__global__ void spin(volatile int* flag) {
    while (!flag[0]) { }
}

int main(int argc, char** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    int nblocks = argc > 2 ? atoi(argv[2]) : -1;
    hipError_t e = hipSetDevice(dev);
    if (e != hipSuccess) { fprintf(stderr, "setdevice fail %d\n", e); return 1; }
    int* flag = nullptr;
    hipMalloc(&flag, sizeof(int));
    hipMemset(flag, 0, sizeof(int));
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, dev);
    if (nblocks < 0) nblocks = p.multiProcessorCount * 16;
    spin<<<nblocks, 256>>>(flag);
    e = hipGetLastError();
    if (e != hipSuccess) { fprintf(stderr, "launch fail %d\n", e); return 1; }
    printf("SPINNING dev=%d blocks=%d\n", dev, nblocks);
    fflush(stdout);
    hipDeviceSynchronize();
    return 0;
}
