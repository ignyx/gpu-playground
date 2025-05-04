# Based on https://dnmtechs.com/using-cmake-in-setup-py-extending-setuptools-for-python-3/

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import platform

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', include_dirs=[]):
        Extension.__init__(self, name, sources=[], include_dirs=include_dirs)
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))) + '/build/'
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == 'Windows':
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        env['CUDAHOSTCXX'] = 'g++-13'
        build_dir=ext.sourcedir + '/build'
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_dir, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_dir)
