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
set(CMAKE_CXX_STANDARD 14)
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

cmake_minimum_required(VERSION 3.26.0)

project(optix7course VERSION 1.0.1
LANGUAGES C CXX CUDA)

if (NOT WIN32)
# visual studio doesn't like these (not need them):
set (CMAKE_CXX_FLAGS "--std=c++17" )
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)
include(${gdt_dir}/cmake/configure_slang.cmake)

#set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
#include(${gdt_dir}/cmake/configure_glfw.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE LEGACY)
if (WIN32)
#  set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
  set(glfw_dir ${PROJECT_SOURCE_DIR}/common/3rdParty/glfw/)
  include_directories(${glfw_dir}/include)
  add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
  find_package(glfw3 REQUIRED)
endif()
include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)


# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

add_subdirectory(renderer)

nanobind_add_module(SampleRenderer NB_STATIC ${PROJECT_SOURCE_DIR}/bindings.cpp)

execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_include_directories(SampleRenderer PRIVATE
  "${TORCH_INCLUDE_DIR}",
  "${TORCH_INCLUDE_DIR}/torch/csrc/api/include"
)

target_link_libraries(SampleRenderer PRIVATE
        slang_stuff
)

nanobind_add_stub(SampleRenderer_stub MODULE SampleRenderer OUTPUT SampleRenderer.pyi MARKER_FILE py.typed DEPENDS SampleRenderer)

install(TARGETS SampleRenderer LIBRARY DESTINATION .)



