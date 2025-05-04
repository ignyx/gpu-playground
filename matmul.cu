#include "./gpuassert.cu"
#include <stdio.h>

// Calculates the dot product sum for a single coefficient.
// Run with a dim3.
__global__ void matmul_elem(const int N, const float *a, const float *b,
                            float *dest) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int column = blockIdx.y * blockDim.y + threadIdx.y;

  if (column < N && row < N) {
    float dot_product = 0.f;

    for (int i = 0; i < N; i++) {
      dot_product += a[row * N + i] * b[i * N + column];
    }
    dest[row * N + column] = dot_product;
    // dest[row * N + column] = (float)column;
  }
}

__host__ void print_square_matrix(const char *name, const float *matrix,
                                  const int N) {
  printf("%s :\n", name);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      const float value = matrix[i * N + j];
      printf("%.1f ", value);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  printf("Starting...\n");

  const int N = 4096;
  const int BLOCK_SIZE = 32; // should be <= 1024 I think

  // data on host
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *sum = new float[N * N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      const int index = i * N + j;
      a[index] = i;
      b[index] = 2 * i + 1;
    }
  }

  // data on device
  float *a_d, *b_d, *sum_d;
  cudaMalloc((void **)&a_d, N * N * sizeof(float));
  cudaMalloc((void **)&b_d, N * N * sizeof(float));
  cudaMalloc((void **)&sum_d, N * N * sizeof(float));

  // copy data from host to device
  cudaMemcpy(a_d, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printf("Copied data to device. Calculating...\n");

  dim3 dimGrid(ceil(N / (float)BLOCK_SIZE), ceil(N / (float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, sum_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  printf("Done calculating, retrieving data and freeing...\n");

  cudaMemcpy(sum, sum_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(sum_d);

  printf("Done. Data :\n");
  /*
  print_square_matrix("a", a, N);
  print_square_matrix("b", b, N);
  print_square_matrix("s", sum, N);
  */

  return 0;
}
