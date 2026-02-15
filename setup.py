from setuptools import setup, Extension
import pybind11

functions_module = Extension(
    'my_framework_cpp',
    sources=['src/bindings.cpp', 'src/tensor.cpp'],
    include_dirs=[pybind11.get_include(), 'src'],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'] 
)

setup(
    name='my_framework_cpp',
    version='1.0',
    ext_modules=[functions_module],
)