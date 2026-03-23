# import os
# from setuptools import setup
# from torch.utils import cpp_extension
# from torch.utils.cpp_extension import BuildExtension
#
# _src_path = os.path.dirname(os.path.abspath(__file__))
#
# setup(
#     name='optix_renderer',
#     description='CUDA RayTracer with BVH acceleration for 3DGS',
#     ext_modules=[
#         cpp_extension.CUDAExtension(
#             name='optix_renderer._C',
#             sources=[os.path.join(_src_path, 'example07_firstRealModel', f) for f in [
#                 'SampleRenderer.cpp',
#                 'bindings.cpp',
#             ]],
#             include_dirs=[
#                 os.path.join(_src_path, 'common'),
#             ],
#             extra_compile_args={
#                 "nvcc": ["-O3", "--expt-extended-lambda"],
#                 "cxx": ["-O3"]}
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension,
#     },
# )
#
# from skbuild import setup  # scikit-build uses setup() from skbuild
# import os
#
# setup(
#     name="optix_renderer",
#     description="CUDA RayTracer with BVH acceleration for 3DGS",
#     cmake_source_dir=".",  # Root directory containing your CMake files
#     cmake_args=[
#         "-DCMAKE_CUDA_FLAGS='-O3 --expt-extended-lambda'",
#     ],
#     build_type="Release",
#     build_args=["--target", "optix_renderer"]  # Specify your actual target
# )
