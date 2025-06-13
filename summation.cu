// Block-level reduction kernel (Kernel 7 from Harris' PDF)

#include <iostream>
#include <cuda_runtime.h>

#define N (2025 * 1024)

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce7(const float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float sum = 0;
    while (i < n) {
        sum += static_cast<float>(g_idata[i]) + static_cast<float>(g_idata[i + blockSize]);
        i += gridSize;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void shortToFloat(const short* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

float run_fast_reduction(short* d_input, float* d_pv_high_float) {
    const int red_blkSize = 512;
    int current_size = 1440 * 1440;

    float *d_input_float, *d_out, *prev_buffer = nullptr;

    // Convert short* to float*
    cudaMalloc(&d_input_float, current_size * sizeof(float));
    shortToFloat<<<(current_size + 255)/256, 256>>>(d_input, d_input_float, current_size);

    while (current_size > 1) {
        int gridSize = (current_size + red_blkSize * 2 - 1) / (red_blkSize * 2);
        cudaMalloc(&d_out, gridSize * sizeof(float));

        reduce7<red_blkSize><<<gridSize, red_blkSize, red_blkSize * sizeof(float)>>>(d_input_float, d_out, current_size);

        if (prev_buffer != nullptr) cudaFree(prev_buffer);
        prev_buffer = d_input_float;
        d_input_float = d_out;
        current_size = gridSize;
    }
    if (prev_buffer != nullptr) {
    cudaFree(prev_buffer);
    }
    // Final result in d_input_float[0]
    cudaMemcpy(d_pv_high_float, d_input_float, sizeof(float), cudaMemcpyDeviceToDevice);
    float result;
    cudaMemcpy(&result, d_pv_high_float, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_float);
    return result;
}
int main() {
    short *h_input = new short[N];

    // Initialize host data once
    for (size_t i = 0; i < N; ++i)
        h_input[i] = 1;

    // Allocate and copy data to device once
    short *d_input;
    float* d_pv_high_float;
    cudaMalloc((void**)&d_pv_high_float, sizeof(float));
    cudaMalloc(&d_input, N * sizeof(short));
    cudaMemcpy(d_input, h_input, N * sizeof(short), cudaMemcpyHostToDevice);

    // Run the kernel multiple times with same input
    for (int j = 0; j < 1000000000; ++j) {
        float result = run_fast_reduction(d_input, d_pv_high_float);
        std::cout << "Run #" << j + 1 << ", Final sum: " << result << std::endl;
    }
    cudaFree(d_pv_high_float);
    // Cleanup
    delete[] h_input;
    cudaFree(d_input);

    return 0;
}


