#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../gpuassert.cu"
#include "numpy/ndarraytypes.h"
#include <Python.h>
#include <numpy/arrayobject.h>

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

/**
 * Multiply two square matrices a*b of dimensions n*n
 *
 * NOTE : not an optimal implementation
 */
static void matmul_gpu(const npy_intp N, const npy_complex128 *a,
                       const npy_complex128 *b, npy_complex128 *dest) {
  printf("[matmul] Allocating and copying to device...\n");

  // data on device
  npy_complex128 *a_d, *b_d, *sum_d;
  cudaMalloc((void **)&a_d, N * N * sizeof(npy_complex128));
  cudaMalloc((void **)&b_d, N * N * sizeof(npy_complex128));
  cudaMalloc((void **)&sum_d, N * N * sizeof(npy_complex128));

  // copy data from host to device
  cudaMemcpy(a_d, a, N * N * sizeof(npy_complex128), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N * N * sizeof(npy_complex128), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printf("[matmul] Copied data to device. Calculating...\n");

  const int BLOCK_SIZE = 32; // because 32**2 = 1024 threads
  dim3 dimGrid(ceil(N / (float)BLOCK_SIZE), ceil(N / (float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, sum_d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  printf("[matmul] Done calculating, retrieving data and freeing...\n");

  cudaMemcpy(dest, sum_d, N * N * sizeof(npy_complex128),
             cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(sum_d);
  printf("[matmul] Done.\n");
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

  matmul_gpu(dimensions[0], data, data, result_matrix);

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
