#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include "hip/hip_runtime.h"

typedef enum
{
    miopenTensorNCHW = 0, // NCHW is the only format supported by miopen
    miopenTensorNHWC = 1 
} miopenTensorFormat_t;

// get hipfunctions
hipFunction_t get_hipfunction(std::string module_path, std::string kernel_name)
{
    hipModule_t module;
    hipModuleLoad(&module, module_path.c_str());
    hipFunction_t hip_func;
    hipModuleGetFunction(&hip_func, module, kernel_name.c_str());
    return hip_func;
}

namespace at
{
    namespace cuda
    {

        namespace utils
        {

            static inline int MaxSharedMemoryPerMultiprocessor(int device_id)
            {
                return getDeviceProperties(device_id)->maxSharedMemoryPerMultiProcessor;
            }
        }
    }
}

#endif
