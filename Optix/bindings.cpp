#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <torch/extension.h>
#include "renderer/SampleRenderer.h"
#include "renderer/helperFunctionsDevice.h"

namespace nb = nanobind;
using namespace osc;

template <typename T>
using NB1DArrayGPU = nb::ndarray<T, nb::shape<-1>, nb::c_contig,
                                 nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_2_3_ArrayGPU = nb::ndarray<T, nb::shape<-1, 2, 3>, nb::c_contig,
                                      nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_3_ArrayGPU = nb::ndarray<T, nb::shape<-1, 3>, nb::c_contig,
                                    nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_4_ArrayGPU = nb::ndarray<T, nb::shape<-1, 4>, nb::c_contig,
                                    nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_9_ArrayGPU = nb::ndarray<T, nb::shape<-1, 9>, nb::c_contig,
                                    nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_1_ArrayGPU = nb::ndarray<T, nb::shape<-1, 1>, nb::c_contig,
                                    nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_N_3_ArrayGPU = nb::ndarray<T, nb::shape<-1, -1, 3>, nb::c_contig,
                                      nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_N_1_ArrayGPU = nb::ndarray<T, nb::shape<-1, -1, 1>, nb::c_contig,
                                      nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_N_N_ArrayGPU = nb::ndarray<T, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cuda, nb::pytorch>;

template <typename T>
using NB_N_N_ArrayGPU = nb::ndarray<T, nb::shape<-1, -1>, nb::c_contig, nb::device::cuda, nb::pytorch>;

NB_MODULE(SampleRenderer, m)
{

      m.def("len",
            [](const NB1DArrayGPU<float> &data) -> size_t
            {
                  return data.shape(0);
            });

      nb::class_<Model>(m, "Model")
          .def(nb::init<>())
          .def_rw("aabbs", &Model::aabbs)
          .def_rw("numEllipsoids", &Model::numEllipsoids)
          .def("assign_tensor_pointers", &Model::assign_tensor_pointers)
          .def("python_side_assign", [](Model &self,
                                        NB_N_2_3_ArrayGPU<float> aabbs,
                                        int numEllipsoids)
               { self.assign_tensor_pointers(aabbs.data(),
                                             numEllipsoids); })
          .def("updateSize", &Model::updateSize);


      m.def("calc_rots_and_aabb",
            [](NB_N_3_ArrayGPU<float> &centers,
               NB_N_3_ArrayGPU<float> &scales,
               NB_N_4_ArrayGPU<float> &rotations,
               NB_N_9_ArrayGPU<float> &rotationMats,
               NB_N_2_3_ArrayGPU<float> &boundingBoxes)
            {
                  convertQuaternionToRotationMatrix(rotations.data(),
                                                    rotationMats.data(),
                                                    rotations.shape(0));
                  calculateBoundingBoxes(centers.data(),
                                         scales.data(),
                                         rotationMats.data(),
                                         boundingBoxes.data(),
                                         centers.shape(0));
            });

      m.def("calc_rot_gradients",
            [](NB_N_4_ArrayGPU<float> &rotation,
               NB_N_4_ArrayGPU<float> &normalized_rotation,
               NB_N_4_ArrayGPU<float> &grad_rotations,
               NB_N_9_ArrayGPU<float> &grad_rotations_mat)
            {
                  calc_rot_gradients(rotation.data(), normalized_rotation.data(), grad_rotations.data(), grad_rotations_mat.data(), rotation.shape(0));
            });

      nb::class_<SampleRenderer>(m, "SampleRenderer")
          .def(nb::init<Model *>())
          .def("setCamera", &SampleRenderer::setCamera)
          .def("setCameraFromIntrinsics", &SampleRenderer::setCameraFromIntrinsics)
          .def("render", &SampleRenderer::render)
          .def("resize", &SampleRenderer::resize)
          .def("getOutputSize", &SampleRenderer::getOutputSize)
          .def("getIntersectionsBufferPointer", &SampleRenderer::getIntersectionsBufferPointer)
          .def("getNumIntersectionsBufferPointer", &SampleRenderer::getNumIntersectionsBufferPointer)
          .def("setDepthBuffer", [](SampleRenderer &self, NB_N_N_1_ArrayGPU<float> &depth){
              size_t h = depth.shape(0);
              size_t w = depth.shape(1);
              self.setDepthBufferDevice(depth.data(), h*w);
          })
          .def("setDepthBuffer", [](SampleRenderer &self, NB_N_N_ArrayGPU<float> &depth){
              size_t h = depth.shape(0);
              size_t w = depth.shape(1);
              self.setDepthBufferDevice(depth.data(), h*w);
          })
          .def("setDepthMaxTolerance", &SampleRenderer::setDepthMaxTolerance)
          .def("downloadIntersectionData", [](const SampleRenderer &self, nb::ndarray<uint32_t> &cpu_buffer, size_t buffer_size) {
              self.downloadIntersectionData(cpu_buffer.data(), buffer_size);
          })
          .def("downloadNumIntersectionsData", [](const SampleRenderer &self, nb::ndarray<uint32_t> &cpu_buffer, size_t buffer_size) {
              self.downloadNumIntersectionsData(cpu_buffer.data(), buffer_size);
          })
          .def_prop_ro("intersections_buffer", [](const osc::SampleRenderer &self) {
            auto dims = self.getOutputSize();
            return NB_N_N_1_ArrayGPU<const float>{self.getIntersectionsBufferPointer(), {static_cast<size_t>(dims.first), static_cast<size_t>(dims.second), 1}, nb::handle(), {}, nb::dtype<const float>(), nb::device::cuda::value, 0};
          }, nb::rv_policy::reference_internal)
          .def("downloadPixels", [](SampleRenderer &self,
                                     nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cpu> &cpu_buffer,
                                     size_t buffer_size) {
              self.downloadPixels(cpu_buffer.data(), buffer_size);
          })
          .def("buildAccel", &SampleRenderer::buildAccel)
          .def("updateParameters",
               [](SampleRenderer &self,
                               NB_N_3_ArrayGPU<float> &centers,
                               NB_N_3_ArrayGPU<float> &scales,
                               NB_N_9_ArrayGPU<float> &rotationMats,
                               NB_N_1_ArrayGPU<float> &densities,
                               int num_ellipsoids,
                               bool allocate_new)
               {
                     self.updateParameters(centers.data(), 
                                          scales.data(), 
                                          rotationMats.data(), 
                                          densities.data(),
                                          num_ellipsoids, 
                                          allocate_new);
               });

      nb::class_<vec3f>(m, "vec3f")
          .def(nb::init<float, float, float>());

      nb::class_<vec2i>(m, "vec2i")
          .def(nb::init<int, int>());

      nb::class_<Camera>(m, "Camera")
          .def(nb::init<vec3f, vec3f, vec3f>());

      m.def("processIntersectionColors",
            [](NB_N_N_N_ArrayGPU<uint32_t> &intersection_ids,
               NB_N_N_ArrayGPU<uint32_t> &num_intersections,
               NB_N_N_3_ArrayGPU<float> &pixel_colors,
               NB_N_3_ArrayGPU<float> &gaussian_colors,
               NB1DArrayGPU<int32_t> &gaussian_hit_counts,
               int max_intersections,
               int num_gaussians)
            {
                  int height = intersection_ids.shape(0);
                  int width = intersection_ids.shape(1);
                  int max_int = intersection_ids.shape(2);
                  if (max_int != max_intersections) {
                        throw std::runtime_error("intersection_ids shape mismatch in max_intersections");
                  }
                  if (num_intersections.shape(0) != height || num_intersections.shape(1) != width) {
                        throw std::runtime_error("num_intersections shape mismatch");
                  }
                  if (pixel_colors.shape(0) != height || pixel_colors.shape(1) != width || pixel_colors.shape(2) != 3) {
                        throw std::runtime_error("pixel_colors shape mismatch");
                  }
                  if (gaussian_colors.shape(0) != num_gaussians || gaussian_colors.shape(1) != 3) {
                        throw std::runtime_error("gaussian_colors shape mismatch");
                  }
                  if (gaussian_hit_counts.shape(0) != num_gaussians) {
                        throw std::runtime_error("gaussian_hit_counts shape mismatch");
                  }
                  processIntersectionColors(
                      reinterpret_cast<uint32_t*>(intersection_ids.data()),
                      reinterpret_cast<uint32_t*>(num_intersections.data()),
                      pixel_colors.data(),
                      gaussian_colors.data(),
                      reinterpret_cast<int32_t*>(gaussian_hit_counts.data()),
                      height, width, max_intersections, num_gaussians);
            });

      m.def("processIntersectionMedianColors",
            [](NB_N_N_N_ArrayGPU<uint32_t> &intersection_ids,
               NB_N_N_ArrayGPU<uint32_t> &num_intersections,
               NB_N_N_3_ArrayGPU<float> &pixel_colors,
               NB_N_3_ArrayGPU<float> &gaussian_median_colors,
               NB1DArrayGPU<int32_t> &gaussian_hit_counts,
               int max_intersections,
               int num_gaussians,
               int max_hits_per_gaussian = 512)
            {
                  int height = intersection_ids.shape(0);
                  int width = intersection_ids.shape(1);
                  int max_int = intersection_ids.shape(2);
                  if (max_int != max_intersections) {
                        throw std::runtime_error("intersection_ids shape mismatch in max_intersections");
                  }
                  if (num_intersections.shape(0) != height || num_intersections.shape(1) != width) {
                        throw std::runtime_error("num_intersections shape mismatch");
                  }
                  if (pixel_colors.shape(0) != height || pixel_colors.shape(1) != width || pixel_colors.shape(2) != 3) {
                        throw std::runtime_error("pixel_colors shape mismatch");
                  }
                  if (gaussian_median_colors.shape(0) != num_gaussians || gaussian_median_colors.shape(1) != 3) {
                        throw std::runtime_error("gaussian_median_colors shape mismatch");
                  }
                  if (gaussian_hit_counts.shape(0) != num_gaussians) {
                        throw std::runtime_error("gaussian_hit_counts shape mismatch");
                  }

                  float *gaussian_color_values;
                  size_t color_values_size = static_cast<size_t>(num_gaussians) * max_hits_per_gaussian * 3 * sizeof(float);
                  cudaMalloc(&gaussian_color_values, color_values_size);
                  cudaMemset(gaussian_color_values, 0, color_values_size);

                  cudaMemset(gaussian_hit_counts.data(), 0, num_gaussians * sizeof(int32_t));

                  processIntersectionMedianColors(
                      reinterpret_cast<uint32_t*>(intersection_ids.data()),
                      reinterpret_cast<uint32_t*>(num_intersections.data()),
                      pixel_colors.data(),
                      gaussian_color_values,
                      gaussian_median_colors.data(),
                      reinterpret_cast<int32_t*>(gaussian_hit_counts.data()),
                      height, width, max_intersections, num_gaussians, max_hits_per_gaussian);

                  cudaFree(gaussian_color_values);
            });
}
