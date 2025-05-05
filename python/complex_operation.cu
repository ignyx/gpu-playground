#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include <Python.h>
#include <numpy/arrayobject.h>

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
  PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(
      input_array, NPY_COMPLEX128,
      NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS);
  if (array == NULL) {
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
  npy_complex128 *data = (npy_complex128 *)PyArray_DATA(array);

  // Perform some operation on the data
  for (npy_intp i = 0; i < size; ++i) {
    // data[i] = data[i] * 2.0; // Example operation: multiply each element by 2
  }

  // Decrease the reference count of the input array
  Py_DECREF(array);

  // Return the modified array
  Py_INCREF(input_array);
  return input_array;
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
