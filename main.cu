#include <cassert>
#include <cmath>
#include <stdio.h>
#define N 1024
#define bytes N * sizeof(int)
__global__ void add(int *a, int *b, int *c) {
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}
int main() {
  int h_a[N], h_b[N], h_c[N];
  int *d_a;
  int *d_b;
  int *d_c;
  cudaError_t cuda_status;
  cuda_status = cudaMalloc(&d_a, bytes);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaMalloc(&d_b, bytes);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaMalloc(&d_c, bytes);
  assert(cuda_status == cudaSuccess);

  for (int i = 0; i < N; i++) {
    h_a[i] = 1;
    h_b[i] = 2;
  }

  cuda_status = cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);

  int NUM_THREADS = 256;
  int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

  add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c);
  cuda_status = cudaGetLastError();
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);

  for (int i = 0; i < N; i++) {
    assert(h_c[i] == h_a[i] + h_b[i]);
    printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
