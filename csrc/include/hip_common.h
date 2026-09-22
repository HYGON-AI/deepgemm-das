// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>


#define HIP_CALL(call)                                                                                                           \
    do                                                                                                                           \
    {                                                                                                                            \
        hipError_t err = call;                                                                                                   \
        if (err != hipSuccess)                                                                                                   \
        {                                                                                                                        \
            printf("\n[deepgemm] %s:%d fail to call %s ---> [HIP error](%s)\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); \
            exit(0);                                                                                                             \
        }                                                                                                                        \
    } while (0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};

struct AiterAsmKernelArgs
{
    void *args_ptr;
    void *arg_size_ptr;
    size_t gdx;
    size_t gdy;
    size_t gdz;
    size_t bdx;
    size_t bdy;
    size_t bdz;
    const hipStream_t stream;
};

class AiterAsmKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    /// Loads \p hsaco from DEEPGEMM_ASM_DIR. If \p name_alt is non-null and non-empty,
    /// tries \p name first, then \p name_alt (same .co may export either symbol across gfx/toolchains).
    AiterAsmKernel(const char *name, const char *hsaco, const char *name_alt = nullptr)
    {
        const char *DEEPGEMM_ASM_DIR = std::getenv("DEEPGEMM_ASM_DIR");
        std::string mod_path = std::string(DEEPGEMM_ASM_DIR ? DEEPGEMM_ASM_DIR : "") + hsaco;
        HIP_CALL(hipModuleLoad(&module, mod_path.c_str()));
        hipError_t err = hipModuleGetFunction(&kernel_func, module, name);
        if (err != hipSuccess && name_alt != nullptr && name_alt[0] != '\0') {
            err = hipModuleGetFunction(&kernel_func, module, name_alt);
        }
        if (err != hipSuccess) {
            printf(
                "\n[deepgemm] %s:%d hipModuleGetFunction failed primary \"%s\"",
                __FILE__, __LINE__, name);
            if (name_alt != nullptr && name_alt[0] != '\0') {
                printf(" fallback \"%s\"", name_alt);
            }
            printf(" ---> [%s]\n", hipGetErrorString(err));
            exit(0);
        }
    };

    ~AiterAsmKernel()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };

    void launch_kernel_ext(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipExtModuleLaunchKernel(kernel_func,
                                       kargs.gdx * kargs.bdx, kargs.gdy * kargs.bdy, kargs.gdz * kargs.bdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config, 0, 0, 0));
    };
};

class AiterAsmKernelFast
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AiterAsmKernelFast(const char *name, void *hsaco)
    {
        HIP_CALL(hipModuleLoadData(&module, hsaco));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        //std::cout << " Success" << std::endl;
    };

    ~AiterAsmKernelFast()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };
};

