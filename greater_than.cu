#include <stdio.h>
#include <cuda.h>
#include <float.h>

#define N (1440 * 1440)
#define THREADS 1024

__global__ void reduce_max_pass1(const float *in, float *out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    float x1 = (i < n) ? in[i] : -FLT_MAX;
    float x2 = (i + blockDim.x < n) ? in[i + blockDim.x] : -FLT_MAX;
    sdata[tid] = fmaxf(x1, x2);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 1]);
    }

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

__global__ void reduce_max_pass2(const float *in, float *out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    if (tid < n)
        sdata[tid] = in[tid];
    else
        sdata[tid] = -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 1]);
    }

    if (tid == 0)
        out[0] = sdata[0];
}

int main() {
    float *h_in = (float*)malloc(N * sizeof(float));
    float *d_in, *d_tmp, *d_out;
    float h_result;

    for (int i = 0; i < N; ++i)
        h_in[i] = (float)i;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS * 2 - 1) / (THREADS * 2);
    cudaMalloc(&d_tmp, blocks * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // First pass
    reduce_max_pass1<<<blocks, THREADS, THREADS * sizeof(float)>>>(d_in, d_tmp, N);

    // Second pass
    reduce_max_pass2<<<1, THREADS, THREADS * sizeof(float)>>>(d_tmp, d_out, blocks);

    cudaMemcpy(&h_result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max value: %f\n", h_result);

    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
