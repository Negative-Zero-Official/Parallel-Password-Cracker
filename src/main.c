#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "kernel.cuh"
#include "../include/utils.h"

unsigned long long getTargetId(const char *target, const char *prefix, const char *charset, int charset_len, int search_len)
{
    const char *suffix = target + strlen(prefix);
    unsigned long long id = 0;
    unsigned long long multiplier = 1;
    for (int i = 0; i < search_len; i++)
    {
        int char_idx = 0;
        while (charset[char_idx] != suffix[i] && char_idx < charset_len)
            char_idx++;
        id += (unsigned long long)char_idx * multiplier;
        multiplier *= (unsigned long long)charset_len;
    }
    return id;
}

void idToString(unsigned long long id, char *res, const char *charset, int charset_len, int search_len, const char *prefix)
{
    int idx = 0;
    int p_len = strlen(prefix);
    for (int i = 0; i < p_len; i++)
        res[idx++] = prefix[i];
    for (int i = 0; i < search_len; i++)
    {
        res[idx++] = charset[id % charset_len];
        id /= charset_len;
    }
    res[idx] = '\0';
}

int main()
{
    printf("=== Parallel Password Cracker ===\n");

    char target[MAX_STR_LEN];
    char prefix[MAX_STR_LEN] = {0}; // Initialize empty
    int min_len, max_len;
    char known_len_ans, known_bday_ans, use_special;

    printf("Enter the password to be matched: ");
    scanf("%255s", target);

    printf("Is the password length known? (y/n): ");
    scanf(" %c", &known_len_ans);

    if (known_len_ans == 'y')
    {
        printf("Enter the length (excluding prefix): ");
        scanf("%d", &min_len);
        max_len = min_len;
    }
    else
    {
        printf("Enter min length: ");
        scanf("%d", &min_len);
        printf("Enter max length: ");
        scanf("%d", &max_len);
    }

    printf("Is the prefix known? (y/n): ");
    scanf(" %c", &known_bday_ans);
    if (known_bday_ans == 'y')
    {
        printf("Enter the known prefix: ");
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
    unsigned long long winning_id = 0;
    int winning_len = 0;

    for (int len = min_len; len <= max_len; ++len)
    {
        printf("Scanning length %d space...\n", len);

        found = launchSearchKernel(target, active_charset, charset_len, len, prefix, maxGridSize, &winning_id);

        if (found)
        {
            winning_len = len;
            break;
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double gpu_secs = milliseconds / 1000.0;

    if (!found)
    {
        printf("\nString was not found in the defined search space.\n");
        return 0;
    }
    printf("GPU Time: %.4f s\n", gpu_secs);
    printf("Running CPU comparison for %.4f seconds...\n", gpu_secs);

    unsigned long long cpu_perms = 0;
    clock_t cpu_start = clock();
    clock_t cpu_limit = (clock_t)(gpu_secs * CLOCKS_PER_SEC);
    char cpu_guess[MAX_STR_LEN];
    int p_len = strlen(prefix);

    while ((clock() - cpu_start) < cpu_limit)
    {
        int cur = 0;
        for (int i = 0; i < p_len; i++)
            cpu_guess[cur++] = prefix[i];
        unsigned long long temp_id = cpu_perms;
        for (int i = 0; i < winning_len; i++)
        {
            cpu_guess[cur++] = active_charset[temp_id % charset_len];
            temp_id /= charset_len;
        }
        cpu_guess[cur] = '\0';

        bool match = true;
        for (int i = 0; i < cur; i++)
        {
            if (cpu_guess[i] != target[i])
            {
                match = false;
                break;
            }
        }

        if (match)
            break;
        cpu_perms++;
    }

    char cpu_last_str[MAX_STR_LEN];
    idToString(cpu_perms, cpu_last_str, active_charset, charset_len, winning_len, prefix);

    printf("CPU reached permutation: %llu (%s)\n", cpu_perms, cpu_last_str);

    // Calculate speedup
    unsigned long long target_dist = getTargetId(target, prefix, active_charset, charset_len, winning_len);
    double cpu_rate = (double)cpu_perms / gpu_secs;
    double est_cpu_time = (double)target_dist / cpu_rate;

    if (cpu_perms == 0)
    {
        est_cpu_time = 0.0;
    }

    printf("\n=== RESULTS ===\n");
    printf("Estimated CPU time for full match: %.4f seconds\n", est_cpu_time);
    printf("Computed Speedup Factor: %.2fx\n", est_cpu_time / gpu_secs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}