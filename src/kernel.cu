#include "kernel.cuh"
#include "../include/utils.h"
#include <stdio.h>
#include <string.h>

// A global flag in device memory to signal all threads to stop if a match is found
__device__ volatile bool d_found_flag = false;

// GPU Kernel
__global__ void stringMatchKernel(
    const char *target,
    const char *charset,
    int charset_len,
    int search_len,
    char *prefix,
    int prefix_len,
    unsigned long long total_permutations
)
{
    // Starting id and stride
    unsigned long long start_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    // If another thread found the match, then return
    if (d_found_flag) return;

    for (unsigned long long id = start_id;
        id < total_permutations;
        id += stride
    ) {
        if (d_found_flag) return;

        char guess[MAX_STR_LEN];
        int current_idx = 0;

        // Prepend the known birthday as prefix
        for (int i=0; i<prefix_len; i++) guess[current_idx++] = prefix[i];

        // Generate permutation based on id
        unsigned long long temp_id = id;
        for (int i=0; i<search_len; i++) {
            guess[current_idx++] = charset[temp_id % charset_len];
            temp_id /= charset_len;
        }

        guess[current_idx] = '\0'; // Null termination

        bool match = true;
        for (int i=0; i<=current_idx; i++) {
            if (guess[i] != target[i]) {
                match = false;
                break;
            }
        }

        if (match) {
            d_found_flag = true;
            printf("\n[GPU THREAD %llu | TEST %llu] Match found: %s\n", start_id, id-start_id, guess);
            return;
        }
    }
}

// Host wrapper to launch kernel
bool launchSearchKernel(
    const char *target,
    const char *charset,
    int charset_len,
    int search_len,
    const char *prefix,
    unsigned long long maxGridSize
)
{
    // Reset the found flag
    bool h_false = false;
    cudaMemcpyToSymbol(d_found_flag, &h_false, sizeof(bool));

    int prefix_len = strlen(prefix);
    
    // Configure CUDA grid
    int threadsPerBlock = 256;
    int blocksPerGrid = 2048;
    
    char *d_target, *d_charset, *d_prefix;
    cudaMalloc(&d_target, MAX_STR_LEN);
    cudaMalloc(&d_charset, 256);
    cudaMalloc(&d_prefix, MAX_STR_LEN);
    
    cudaMemcpy(d_target, target, MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_charset, charset, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, prefix_len + 1, cudaMemcpyHostToDevice);

    // Calculate total permutations needed (charset size ^ search length)
    unsigned long long total_permutations = 1;
    for (int i=0; i<search_len; i++) total_permutations *= charset_len;

    // Launch kernel
    stringMatchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_target, d_charset, charset_len, search_len, d_prefix, prefix_len, total_permutations);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_target); cudaFree(d_charset); cudaFree(d_prefix);
        return false;
    }

    cudaDeviceSynchronize(); // Wait for GPU to finish

    bool h_found_flag;
    cudaMemcpyFromSymbol(&h_found_flag, d_found_flag, sizeof(bool));

    cudaFree(d_target); cudaFree(d_charset); cudaFree(d_prefix);

    return h_found_flag;
}

// Max Grid Size helper
unsigned long long getMaxGridSize() {
    int deviceid;
    cudaGetDevice(&deviceid);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceid);
    return prop.maxGridSize[0];
}