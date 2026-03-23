# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)
slang_compile_and_embed(embedded_ptx_code2 ${PROJECT_SOURCE_DIR}/renderer/devicePrograms.slang
)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

add_library(slang_stuff SHARED
        ${embedded_ptx_code}
        ${embedded_ptx_code2}
        helperFunctionsDevice.cu
        helperFunctionsDevice.h
        optix7.h
        CUDABuffer.h
        LaunchParams.h
        SampleRenderer.h
        SampleRenderer.cpp
        Model.h
        Model.cpp
)

target_link_libraries(slang_stuff
        PUBLIC
        gdt
        # optix dependencies, for rendering
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        # glfw and opengl, for display
        glfWindow
        glfw
        ${OPENGL_gl_LIBRARY}
)

add_executable(run_renderer
        main.cpp
)

target_link_libraries(run_renderer slang_stuff)

