#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../gpuassert.cu"
#include "numpy/ndarraytypes.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

// Calculates the dot product sum for a single complex coefficient.
// Run with a dim3.
__global__ void matmul_elem(const npy_intp N, const npy_complex128 *a,
                            const npy_complex128 *b, npy_complex128 *dest) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int column = blockIdx.y * blockDim.y + threadIdx.y;

  if (column < N && row < N) {
    double dot_product_real = 0.f;
    double dot_product_imag = 0.f;

    for (int i = 0; i < N; i++) {
      const npy_complex128 a_i = a[row * N + i];
      const npy_complex128 b_i = b[i * N + column];

      dot_product_real += a_i._Val[0] * b_i._Val[0] - a_i._Val[1] * b_i._Val[1];
      dot_product_imag += a_i._Val[0] * b_i._Val[1] + a_i._Val[1] * b_i._Val[0];
    }

    dest[row * N + column]._Val[0] = dot_product_real;
    dest[row * N + column]._Val[1] = dot_product_imag;
  }
}

// Calculates the dot product sum for a single complex coefficient.
// Run with a dim1. Assumes b is diagonal.
// PERF : Could probably benefit from better caching by idexing by column
// instead of row, so the destination matrix would be filled row by row.
__global__ void matmul_diag_elem(const npy_intp N, const npy_complex128 *a,
                                 const npy_complex128 *b,
                                 npy_complex128 *dest) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N) {
    for (int column = 0; column < N; column++) {
      const npy_complex128 a_i = a[row * N + column];
      const npy_complex128 b_i = b[column * N + column];

      double dot_product_real =
          a_i._Val[0] * b_i._Val[0] - a_i._Val[1] * b_i._Val[1];
      double dot_product_imag =
          a_i._Val[0] * b_i._Val[1] + a_i._Val[1] * b_i._Val[0];

      dest[row * N + column]._Val[0] = dot_product_real;
      dest[row * N + column]._Val[1] = dot_product_imag;
    }
  }
}

/**
 * Multiply A * D * Ainv square matrices of dimensions N * N
 *
 * D is a diagonal matrix.
 * Ainv is the inverse matrix of A.
 *
 * NOTE : not an optimal implementation
 */
static void matmul_ADAinv_gpu(const npy_intp N, const npy_complex128 *a,
                              const npy_complex128 *d,
                              const npy_complex128 *ainv,
                              npy_complex128 *dest) {
  printf("[matmul] Allocating and copying to device...\n");
  clock_t start = clock();

  // data on device
  npy_complex128 *a_d, *d_d, *ainv_d, *sum_d;
  cudaMalloc((void **)&a_d, N * N * sizeof(npy_complex128));
  cudaMalloc((void **)&d_d,
             N * N * sizeof(npy_complex128)); // could just be a vec
  cudaMalloc((void **)&ainv_d, N * N * sizeof(npy_complex128));
  cudaMalloc((void **)&sum_d, N * N * sizeof(npy_complex128));

  // copy data from host to device
  cudaMemcpy(a_d, a, N * N * sizeof(npy_complex128), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d, N * N * sizeof(npy_complex128), cudaMemcpyHostToDevice);
  cudaMemcpy(ainv_d, ainv, N * N * sizeof(npy_complex128),
             cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  clock_t end_alloc = (clock() - start) * 1000 / CLOCKS_PER_SEC;
  printf("[matmul] Copied data to device. Calculating...\n");

  // calculate sum = A * D
  const int BLOCK_SIZE_DIAG = 1024; // should be <= 1024 I think
  matmul_diag_elem<<<ceil(N / (float)BLOCK_SIZE_DIAG), BLOCK_SIZE_DIAG>>>(
      N, a_d, d_d, sum_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  clock_t end_diag = (clock() - start) * 1000 / CLOCKS_PER_SEC;

  // calculate A = sum * Ainv
  const int BLOCK_SIZE = 32; // because 32**2 = 1024 threads
  dim3 dimGrid(ceil(N / (float)BLOCK_SIZE), ceil(N / (float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  matmul_elem<<<dimGrid, dimBlock>>>(N, sum_d, ainv_d, a_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  clock_t end_mul = (clock() - start) * 1000 / CLOCKS_PER_SEC;

  printf("[matmul] Done calculating, retrieving data and freeing...\n");

  // copy data from devices to host
  cudaMemcpy(dest, a_d, N * N * sizeof(npy_complex128), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(d_d);
  cudaFree(ainv_d);
  cudaFree(sum_d);

  clock_t end_free = (clock() - start) * 1000 / CLOCKS_PER_SEC;
  printf("[matmul] Done. alloc=%ldms, diag=%ldms, mul=%ldms, free=%ldms\n",
         end_alloc, end_diag, end_mul, end_free);
}

// Function to perform an operation on a complex NumPy array
static PyObject *complex_operation(PyObject *self, PyObject *args) {
  PyObject *input_array;

  // Parse the input arguments
  if (!PyArg_ParseTuple(args, "O", &input_array)) {
    return NULL;
  }

  // Ensure the input is a NumPy array
  if (!PyArray_Check(input_array)) {
    PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
    return NULL;
  }

  // Get the array and ensure it is of complex type
  // increases refcount to array.
  PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(
      input_array, NPY_COMPLEX128,
      NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS);
  if (array == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Input array must contain complex128 values");
    return NULL;
  }

  // Get the dimensions of the array
  int dimension_count = PyArray_NDIM(array);
  npy_intp *dimensions = PyArray_DIMS(array);
  npy_intp size = PyArray_SIZE(array);

  // Ensure the matrix is square
  if (dimension_count != 2 || dimensions[0] != dimensions[1]) {
    PyErr_SetString(PyExc_TypeError, "Input must be a square matrix");
    return NULL;
  }

  // Get a pointer to the data
  const npy_complex128 *data = (npy_complex128 *)PyArray_DATA(array);

  // Perform some operation on the data
  for (npy_intp i = 0; i < size; ++i) {
    // data[i] = data[i] * 2.0; // Example operation: multiply each element by 2
  }

  PyObject *result_matrix_object =
      PyArray_NewLikeArray(array, NPY_CORDER, NULL, 1);
  const PyArrayObject *result_matrix_array =
      (PyArrayObject *)result_matrix_object;
  // Get a pointer to the data
  npy_complex128 *result_matrix =
      (npy_complex128 *)PyArray_DATA(result_matrix_array);
  for (npy_intp i = 0; i < size; ++i) {
    result_matrix[i]._Val[0] = 2.0;
    result_matrix[i]._Val[1] = 3.5;
  }

  matmul_ADAinv_gpu(dimensions[0], data, data, data, result_matrix);

  // printf("Refcount to input_array before free: %d\n",
  // Py_REFCNT(input_array)); printf("Refcount to array before free: %d\n",
  // Py_REFCNT(array));

  // Decrease the reference count of the input array
  Py_DECREF(array);
  return result_matrix_object;
}

// Method definitions
static PyMethodDef ComplexMethods[] = {
    {"complex_operation", complex_operation, METH_VARARGS,
     "Perform an operation on a complex NumPy array"},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef complexmodule = {
    PyModuleDef_HEAD_INIT, "complexmodule",
    "Module for performing operations on complex NumPy arrays", -1,
    ComplexMethods};

// Module initialization
PyMODINIT_FUNC PyInit_complexmodule(void) {
  PyObject *m;
  import_array(); // Initialize NumPy API
  m = PyModule_Create(&complexmodule);
  return m;
}
