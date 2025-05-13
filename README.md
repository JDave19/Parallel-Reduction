# CUDA Parallel Reduction Optimization Summary
Author: Janak Dave

Source: CUDA Webinar by Mark Harris (NVIDIA)  

## 🚀 Overview

Parallel reduction is a fundamental operation in GPU computing. This document summarizes **seven optimized CUDA kernels** for reduction and explains the **underlying optimization concepts** such as warp divergence, shared memory bank conflicts, and memory coalescing.

In CUDA, **warping** and **reduction** are two fundamentally different concepts, though they often interact when writing efficient GPU code.

---

### 🔁 Warping
- **Definition**: In CUDA, a *warp* is a group of **32 threads** that execute the same instruction at the same time (SIMT: Single Instruction, Multiple Threads).
- **Purpose**: It's a hardware-level scheduling unit.
- **Key points**:
  - Warps are managed by the GPU scheduler.
  - Threads in a warp execute in lockstep (unless there’s divergence like an `if-else`).
  - Optimizing warp-level operations (e.g., avoiding divergence) can improve performance.

**Example:**
```cpp
int laneId = threadIdx.x % 32;  // Lane within the warp
```

---

### ➗ Reduction
- **Definition**: Reduction is an algorithmic technique to **combine values across threads**, like computing a sum, min, max, etc.
- **Purpose**: Efficient parallel aggregation of data.
- **Often implemented using warps** for efficiency.

**Example: summing an array in parallel using reduction:**
```cpp
__shared__ float sdata[32];
int tid = threadIdx.x;
sdata[tid] = input[tid];
__syncthreads();

// Simple reduction within block
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

---

### 🚀 Relationship Between Warping and Reduction:
- You can use **warp-level primitives** like `__shfl_down_sync` for faster reduction within a warp (no shared memory needed).
- Example:
  ```cpp
  float val = ...;  // Each thread has a value
  for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  ```

---

### In short:
| Feature     | Warping                           | Reduction                                |
|-------------|-----------------------------------|------------------------------------------|
| What it is  | Hardware thread execution model   | Algorithm to combine values              |
| Involves    | 32 threads per warp               | Multiple threads (in block or grid)      |
| Purpose     | Scheduling unit                   | Data aggregation (e.g., sum, max)        |
| Related?    | Yes, reduction is often optimized using warp-level techniques                |

Summary of **7 parallel reduction kernels** and the related CUDA optimization concepts like **warp divergence**, **bank conflicts**, and **memory coalescing**:


## 🔧 Kernel Versions and Optimizations

### 1. **Interleaved Addressing with Divergent Branching**
```cpp
if (tid % (2*s) == 0)
    sdata[tid] += sdata[tid + s];
```

* **Problem:** Divergent branches due to `%` operator.
* **Drawback:** Warp divergence and slow modulus operations.
### 📌 Warp Divergence

* **What:** Threads in a warp take different execution paths.
* **Why bad:** Causes serialization → reduced parallel efficiency.

---

### 2. **Interleaved Addressing with Strided Indexing**

```cpp
int index = 2 * s * tid;
if (index < blockDim.x)
    sdata[index] += sdata[index + s];
```

* **Fixes divergence** with uniform indexing.
* **Problem:** Causes **shared memory bank conflicts**.
### 📌 Shared Memory Bank Conflicts

* **What:** Multiple threads access the same memory bank.
* **Why bad:** Accesses are serialized → reduced memory bandwidth.
---

### 3. **Sequential Addressing**

```cpp
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];
}
```

* Threads access consecutive addresses.
* **Fixes bank conflicts**.
* **Issue** Not using full memory bandwidth of GPU.

---

### 4. **First Add During Global Load**

```cpp
unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
sdata[threadIdx.x] = g_idata[i] + g_idata[i + blockDim.x];;
```

In Kernel 3, each thread would:
* Load one element from global memory.
* Store it in shared memory.
* Then a series of reductions would happen inside shared memory.

This results in:
1. load per thread
2. A large number of threads, even for simple reductions
3. Each thread loads only one value, causing underutilization of memory bandwidth

**Key idea:**
Each thread loads two elements from global memory, adds them immediately, and stores the result into shared memory.
---

### 5. **Unroll Last Warp**

```cpp
if (tid < 32) {
    warpReduce(sdata, tid); // No __syncthreads needed
}
```

In Kernel 4 (and earlier), as the reduction progresses:
* The number of **active threads** is halved at each iteration.
* Eventually, the last **32 threads (1 warp)** do the final reductions.

But within a warp, **all threads run in lockstep (SIMT)**. So calling __syncthreads() and using if (tid < s) for values of s <= 32 becomes unnecessary — it adds overhead without benefit.

**Omptimzation** 
Once the reduction reaches s <= 32, we:
1. Skip the loop
2. Manually unroll the operations

```cpp
if (tid < 32) {
    volatile int* vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}
```
---

### 6. **Completely Unrolled Reduction (Compile-Time)**

```cpp
template <unsigned int blockSize>
__global__ void reduce6(...) {
    ...
    if (blockSize >= 512) ...
    if (blockSize >= 256) ...
    if (tid < 32) warpReduce<blockSize>(...);
}
```
Even though Kernel 5 unrolls the last warp, the earlier reduction steps **(s = 256, 128, 64, etc.)** are still inside a loop.

Loops add:
1. Branching
2. Index calculations
3. __syncthreads() calls

✅ Optimization in **Kernel 6**

Use C++ templates to completely unroll all reduction steps at compile time, for a known block size.

* Fully unrolls the loop using C++ templates.
* **Compile-time specialization** yields very fast code.

---

### 7. **Multiple Elements Per Thread**

```cpp
while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
}
```

* Each thread processes **multiple input elements**.
* Reduces number of threads and **improves coalescing**.
### 📌 Memory Coalescing

* **What:** Threads access consecutive memory locations.
* **Why good:** Accesses can be grouped into a single memory transaction.



## ✅ Key Takeaways

* **Avoid divergence** and **bank conflicts**.
* Perform more work per thread when possible.
* **Unroll loops** to reduce instruction overhead.
* Use **templates** for flexible, compile-time optimized kernels.

---


### 🔹 What is Template Instantiation?

- A **template** is a C++ blueprint for functions or classes.
- **Instantiation** is when the **compiler generates real code** from that blueprint using specific types or values.

#### ✅ Example:

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}

int x = add<int>(2, 3);       // instantiates add<int>
float y = add<float>(1, 2);   // instantiates add<float>
```

The compiler generates:
```cpp
int add(int, int);        // at compile time
float add(float, float);  // at compile time
```

---

### 🔹 Template Instantiation in CUDA

Templates in CUDA are often used to **optimize kernel performance**:

```cpp
template <unsigned int blockSize>
__global__ void reduce7(...) { ... }

reduce7<512><<<gridSize, 512, ...>>>();  // compile-time instantiation
```

- `blockSize` is a **compile-time constant**.
- The compiler generates a version of `reduce7` for `blockSize = 512`.
- Enables optimizations like **loop unrolling** and **removing unused branches**.

---

### 🔹 Compile Time vs Runtime

| Concept        | Compile Time                              | Runtime (Real Time)                         |
|----------------|-------------------------------------------|---------------------------------------------|
| Happens When?  | While compiling code into a binary        | While the program is running                |
| Values Known?  | Yes (constants, templates, constexpr)     | No (user input, file I/O, dynamic values)   |
| Optimizations  | Loop unrolling, dead code elimination     | Can't optimize as aggressively              |
| Example        | `const int x = 2 + 2;`                    | `int x; std::cin >> x;`                     |

#### ✅ Compile-Time Example:
```cpp
template<int N>
int square() {
    return N * N; // Compiler knows N
}
```

#### ❌ Runtime Example:
```cpp
int square(int n) {
    return n * n; // n unknown until runtime
}
```

---

### 🔹 Why Templates Are Faster Than `if` Conditions

| Feature          | Template Instantiation              | `if` Condition at Runtime             |
|------------------|-------------------------------------|---------------------------------------|
| Evaluated When?  | Compile time                        | Runtime                               |
| Optimized Code?  | Fully (no branches left behind)     | May leave all branches in the binary  |
| Unused Code?     | Removed                             | Must remain just in case              |
| Ideal For        | High-performance GPU kernels        | Dynamic or input-dependent logic      |

---

### ✅ Summary

- Use **templates** when you want **compile-time specialization**.
- Use **`if` conditions** when values are only known at **runtime**.
- Templates + CUDA = highly optimized, unrolled, branchless code.

---
