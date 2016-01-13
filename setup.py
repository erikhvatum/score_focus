#!/usr/bin/env python3

# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Zach Pincus

import distutils.core
import numpy
import pathlib
import subprocess
import sys

extra_compile_args = []
extra_link_args = []

#extra_compile_args.extend(subprocess.run(['pkg-config', 'opencv', '--cflags'], stdout=subprocess.PIPE).stdout.decode('utf8').split())
#extra_link_args.extend(subprocess.run(['pkg-config', 'opencv', '--libs'], stdout=subprocess.PIPE).stdout.decode('utf8').split())

if sys.platform != 'win32':
    extra_compile_args.append('-std=c++11')
    extra_compile_args.append('-march=native')
    extra_compile_args.append('-fopenmp')

try:
    from Cython.Build import cythonize
    ext_processor = cythonize
except ImportError:
    def uncythonize(extensions, **_ignore):
        for extension in extensions:
            sources = []
            for src in map(pathlib.Path, extension.sources):
                if src.suffix == '.pyx':
                    if extension.language == 'c++':
                        ext = '.cpp'
                    else:
                        ext = '.c'
                    src = src.with_suffix(ext)
                sources.append(str(src))
            extension.sources[:] = sources
        return extensions
    ext_processor = uncythonize

_cppmod = distutils.core.Extension(
    '_cppmod',
    language = 'c++',
    sources = [
        'cppmod/_cppmod.pyx',
        'cppmod/_image_stack_median.cpp'],
    depends = ['cppmod/_image_stack_median.h'],
    extra_compile_args = extra_compile_args,
    extra_link_args = ['-fopenmp'],
    include_dirs = [numpy.get_include()])

distutils.core.setup(
    classifiers = [
        'Intended Audience :: Science/Research',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython'],
    ext_modules = ext_processor([_cppmod]),
    description = 'analysis cpp module',
    name = 'analysis',
    packages = ['analysis'],
    version = '1.0')
