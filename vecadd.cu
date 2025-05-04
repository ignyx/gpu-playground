#include <stdio.h>

__global__ void vecadd(const int n, const float *a, const float *b,
                       float *dest) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    dest[i] = a[i] + b[i];
  }
}

__host__ void printVector(const char *name, float *vector, int count) {
  printf("%s :", name);
  for (int i = 0; i < count; i++)
    printf(" %f", vector[i]);
  printf("\n");
}

int main() {
  printf("Starting...\n");

  const int N = 4096;

  // data on host
  float *a = new float[N];
  float *b = new float[N];
  float *sum = new float[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2 * i + 1;
  }

  // data on device
  float *a_d, *b_d, *sum_d;
  cudaMalloc((void **)&a_d, N * sizeof(float));
  cudaMalloc((void **)&b_d, N * sizeof(float));
  cudaMalloc((void **)&sum_d, N * sizeof(float));

  // copy data from host to device
  cudaMemcpy(a_d, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N * sizeof(float), cudaMemcpyHostToDevice);

  printf("Copied data to device. Calculating...\n");

  const int BLOCK_SIZE = 256; // should be <= 1024 I think
  vecadd<<<ceil(N / (float)BLOCK_SIZE), BLOCK_SIZE>>>(N, a_d, b_d, sum_d);

  printf("Done calculating, retrieving data and freeing...\n");

  cudaMemcpy(sum, sum_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(sum_d);

  printf("Done. Data :\n");
  const int count = 5;
  printVector("a", a, count);
  printVector("b", b, count);
  printVector("s", sum, count);

  return 0;
}
