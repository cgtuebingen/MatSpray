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

#include "gdt/math/AffineSpace.h"
#include <vector>
#include "CUDABuffer.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;

  struct vec3x3 {
    vec3f rows[3];  // Each row of the 3x3 matrix is a vec3f

    vec3x3() : rows{vec3f(), vec3f(), vec3f()} {}
    vec3x3(vec3f r0, vec3f r1, vec3f r2) {
        rows[0] = r0;
        rows[1] = r1;
        rows[2] = r2;
    }
};
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f              diffuse;
  };

  struct PerRayData {
    vec3f hitpoint;
    float transmittance;
    float curDensity;
    int curNumOfVolumes;
    bool noHit;
  };

  struct aabb {
    vec3f lower;
    vec3f upper;
  };

  struct Ellipsoid {
    vec3f center;
    vec3f scale;
    float rotation[3][3];
    float opacity;
    aabb boundingBox;
  };

 struct Model {
    float *aabbs;
    int numEllipsoids = 0;

    // Declare the function that assigns tensor pointers
    void assign_tensor_pointers(float *aabbs,
                                int numEllipsoids);
    void updateSize(int newSize);
  };
}
