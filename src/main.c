#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <cuda_runtime_api.h>
#include "kernel.cuh"
#include "../include/utils.h"

int main() {
    printf("=== Parallel Password Cracker ===\n");

    char target[MAX_STR_LEN];
    char prefix[MAX_STR_LEN] = {0}; // Initialize empty
    int min_len, max_len;
    char known_len_ans, known_bday_ans, use_special;

    printf("Enter the password to be matched: ");
    scanf("%255s", target);

    printf("Is the password length known? (y/n): ");
    scanf(" %c", &known_len_ans);

    if (known_len_ans == 'y') {
        printf("Enter the length (excluding birthday prefix): ");
        scanf("%d", &min_len);
        max_len = min_len;
    } else {
        printf("Enter min length: ");
        scanf("%d", &min_len);
        printf("Enter max length: ");
        scanf("%d", &max_len);
    }

    printf("Is the birthday a prefix? (y/n): ");
    scanf(" %c", &known_bday_ans);
    if (known_bday_ans == 'y') {
        printf("Enter the birthday/prefix: ");
        scanf("%255s", prefix);
    }

    printf("Use special characters? (y/n): ");
    scanf(" %c", &use_special);
    
    const char *active_charset = (use_special == 'y') ? CHARSET_SPECIAL : CHARSET_ALPHANUMERIC;
    int charset_len = strlen(active_charset);

    unsigned long long maxGridSize = getMaxGridSize();

    printf("\nStaring parallel search on GPU...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    bool found = false;

    for (int len = min_len; len <= max_len; ++len) {
        printf("Scanning length %d space...\n", len);

        found = launchSearchKernel(target, active_charset, charset_len, len, prefix, maxGridSize);

        if (found) break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (!found) {
        printf("\nString was not found in the defined search space.\n");
    }
    printf("Time taken: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}