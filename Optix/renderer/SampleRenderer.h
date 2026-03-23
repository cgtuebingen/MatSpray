// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "cuda_runtime.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

namespace osc {

  struct Camera {
    vec3f from;
    vec3f at;
    vec3f up;
  };
  
  class SampleRenderer
  {
  public:
    SampleRenderer(Model *model);

    std::pair<int, int> getOutputSize() const { return {launchParams.frame.size.y, launchParams.frame.size.x}; }

    void render();

    void resize(const vec2i &newSize);

    void downloadPixels(float *cpu_buffer, size_t buffer_size);

    void setCamera(const Camera &camera);

    void setCameraFromIntrinsics(const Camera &camera,
                                 float focalLengthX, float focalLengthY,
                                 float principalPointX, float principalPointY);

    uint32_t* getIntersectionsBufferPointer() const { return reinterpret_cast<uint32_t*>(intGaussianIdsBuffer.d_ptr); }
    uint32_t* getNumIntersectionsBufferPointer() const { return reinterpret_cast<uint32_t*>(numIntersectionsBuffer.d_ptr); }

    void setDepthBufferDevice(float *d_depth, size_t num_values);
    void setDepthMaxTolerance(float tol) { launchParams.depth_max_tolerance = tol; }
    
    void downloadIntersectionData(uint32_t* cpu_buffer, size_t buffer_size) const;
    void downloadNumIntersectionsData(uint32_t* cpu_buffer, size_t buffer_size) const;

    uint32_t getNumEllipsoids() const { return model->numEllipsoids; }

    void updateParameters(float *centers,
                          float *scales,
                          float *rotationMats,
                          float *densities,
                          const int num_ellipsoids,
                          const bool allocate_new);

    OptixTraversableHandle buildAccel();

  protected:
    void initOptix();
    void createContext();
    void createModule();
    void createRaygenPrograms();
    void createMissPrograms();
    void createHitgroupPrograms();
    void createPipeline();
    void buildSBT();

  protected:
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;

    OptixDeviceContext optixContext;

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModule                 intersection_module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;

    CUDABuffer meansBuffer;
    CUDABuffer rotationsBuffer;
    CUDABuffer scalesBuffer;
    CUDABuffer densitiesBuffer;

    CUDABuffer intGaussianIdsBuffer;
    CUDABuffer numIntersectionsBuffer;
    CUDABuffer prev_indexBuffer;
    CUDABuffer depthBuffer;
    CUDABuffer colorBuffer;
    
    Camera lastSetCamera;
    
    bool usingIntrinsics;
    float lastFx, lastFy, lastCx, lastCy;
    
    Model *model;

    CUDABuffer aabbBufferSingle;
    CUDABuffer numEllipsoidsBuffer;
    CUDABuffer asBuffer;
  };

} // ::osc
