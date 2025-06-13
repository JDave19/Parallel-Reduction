#include <stdio.h>
#include <cuda.h>
#include <climits>
#include <stdlib.h>

#define N 1440 * 1440  // Total elements

template <unsigned int blockSize>
__global__ void max_pass(int *input, int *output, const int n) {
    extern __shared__ int shared_max[];

    int tid = threadIdx.x;
    int global_i = 2 * (blockIdx.x * blockDim.x + tid);

    int val1 = (global_i < n) ? input[global_i] : INT_MIN;
    int val2 = (global_i + 1 < n) ? input[global_i + 1] : INT_MIN;

    shared_max[tid] = max(val1, val2);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        __syncthreads();
    }

    if (blockSize >= 64 && tid < 32) {
        volatile int* vsmem = shared_max;
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 1]);
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_max[0];
    }
}

__global__ void sum_pass(int* input, float *output, int n) {
    __shared__ float partial_sum[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float val = (i < n) ? (float)input[i] : 0.0f;
    partial_sum[tid] = val;
    __syncthreads();

    // Parallel reduction (sum)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            partial_sum[tid] += partial_sum[tid + s];
        __syncthreads();
    }

    // One thread from each block writes partial sum
    if (tid == 0) {
        atomicAdd(output, partial_sum[0]);
    }
}

int main() {
    int *arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        arr[i] = 1;
    }

    int *d_arr, *d_out;
    float *d_mean;
    float h_mean;

    int threadsPerBlock = 1024;
    int elementsPerBlock = threadsPerBlock * 2;
    int blocks = (N + elementsPerBlock - 1) / elementsPerBlock;
    printf("%d ", blocks);

    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_out, blocks * sizeof(int));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMemset(d_mean, 0, sizeof(float));  // Initialize mean to 0

    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    max_pass<1024><<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, d_out, N);

    // Sum the intermediate max values from each block and compute mean
    sum_pass<<<(blocks + 1023) / 1024, 1024>>>(d_out, d_mean, blocks);

    cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    h_mean /= blocks;

    printf("Average of block-wise max = %f\n", h_mean);

    cudaFree(d_arr);
    cudaFree(d_out);
    cudaFree(d_mean);
    free(arr);
    return 0;
}
