#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper function called by the CPU to launch the GPU kernel
bool launchSearchKernel(
    const char *target,
    const char *charset,
    int charset_len,
    int search_len,
    const char *prefix,
    unsigned long long maxGridSize
);

unsigned long long getMaxGridSize();

#ifdef __cplusplus
}
#endif

#endif // KERNEL_CUH