#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include "hip/hip_runtime.h"

#define VERBOSE_DEFAULT false
typedef enum
{
    miopenTensorNCHW = 0, // NCHW is the only format supported by miopen
    miopenTensorNHWC = 1
} miopenTensorFormat_t;

static void processMiopenStatus(const miopenStatus_t &status,
                        const std::string &string = std::string(),
                        bool verbose = VERBOSE_DEFAULT)
{
    if (status != miopenStatusSuccess)
    {
        LOG(FATAL) << string << " " << miopenGetErrorString(status);
    }
    else if (verbose)
    {
        LOG(INFO) << string << " " << miopenGetErrorString(status);
    }
}

static void checkHipStatus(const std::string &string = std::string(),
                     bool verbose = VERBOSE_DEFAULT)
{
    hipError_t status = hipGetLastError();
    if (status != hipSuccess)
    {
        LOG(FATAL) << string << " " << hipGetErrorString(status);
    }
    else if (verbose)
    {
        LOG(INFO) << string << " " << hipGetErrorString(status);
    }
}

// get hipfunctions
static hipFunction_t get_hipfunction(std::string module_path, std::string kernel_name)
{
    hipModule_t module;
    hipModuleLoad(&module, module_path.c_str());
    checkHipStatus("get_hipfunction:module");
    hipFunction_t hip_func;
    hipModuleGetFunction(&hip_func, module, kernel_name.c_str());
    checkHipStatus("get_hipfunction:hip_func");
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
