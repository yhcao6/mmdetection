from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bbox_overlaps_cuda',
    ext_modules=[
        CUDAExtension('bbox_overlaps_cuda', [
            'src/bbox_overlaps_cuda.cpp',
            'src/bbox_overlaps_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
