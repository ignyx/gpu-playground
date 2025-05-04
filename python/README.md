# CUDA in Python

(Work in progress).

This example uses CUDA to accelerate calculation performed on big matrices in a project by friendly
applied math students.

## Building

Requires `Python.h` and numpy headers. Also CUDA, cmake, your generic C toolchain.

```bash
sudo apt-get python3-dev python3-numpy-dev
mkdir build
cd build
cmake ..
make
cd ..
mv build/libcomplex_operation.so ./complexmodule.so
```

`complexmodule` should then be importable. Try running `python3 test.py`.
