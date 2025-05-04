from setuptools import setup, Extension
from cmake_setuptools import CMakeExtension, CMakeBuild
import numpy as np

# Define the extension module
extension_mod = Extension(
# extension_mod = CMakeExtension(
    'complexmodule',
    sources=['complex_operation.cu'],
    include_dirs=[np.get_include()]
)

# Setup configuration
setup(
    name='complexmodule',
    version='1.0',
    description='Module for performing operations on complex NumPy arrays',
    ext_modules=[extension_mod],
    # cmdclass=dict(build_ext=CMakeBuild),
)
