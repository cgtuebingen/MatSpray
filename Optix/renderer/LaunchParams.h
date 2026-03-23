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

#include "Model.h"
#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc
{
  using namespace gdt;

  template <typename T>
  struct StructuredBuffer
  {
    T *data;
    size_t size;
  };

  struct GaussianSBTData
  {
    StructuredBuffer<float> centers;
    StructuredBuffer<float> scales;
    StructuredBuffer<float> rotations;
    StructuredBuffer<float> opacities;
    int numEllipsoids;
  };

  struct LaunchParams
  {
    // Model parameters
    StructuredBuffer<float> centers;
    StructuredBuffer<float> scales;
    StructuredBuffer<float> rotations;
    StructuredBuffer<float> densities;

    // Rendering parameters
    StructuredBuffer<uint32_t> intGaussianIdsBuffer;
    StructuredBuffer<uint32_t> numIntersectionsBuffer;
    StructuredBuffer<uint32_t> prev_indexBuffer;
    StructuredBuffer<float> depthBuffer; // per-pixel max ray distance (t)

    struct
    {
      vec2i size;
    } frame;

    struct
    {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    float epsilon;
    uint32_t max_num_intersections;
    float transmittance_threshold;
    uint32_t numEllipsoids;
    float depth_max_tolerance; // allowable overshoot beyond depth for gating

    OptixTraversableHandle traversable;
  };

} // ::osc
