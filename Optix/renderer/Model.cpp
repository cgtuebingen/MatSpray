// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
//std
#include <set>

namespace tinyobj {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  


  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
      mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }

  void computeGaussianAABB(aabb& bounds, const vec3f& center,
                           const vec3f& scale, const float (&rotation)[3][3]) {
    bounds.lower = vec3f(100000.f);
    bounds.upper = vec3f(-100000.f);

    vec3f sa = scale.x * vec3f(rotation[0][0], rotation[1][0], rotation[2][0]);
    vec3f sb = scale.y * vec3f(rotation[0][1], rotation[1][1], rotation[2][1]);
    vec3f sc = scale.z * vec3f(rotation[0][2], rotation[1][2], rotation[2][2]);

    vec3f corners[8];
    corners[0] = center + sa + sb + sc;
    corners[1] = center + sa + sb - sc;
    corners[2] = center + sa - sb + sc;
    corners[3] = center + sa - sb - sc;
    corners[4] = center - sa + sb + sc;
    corners[5] = center - sa + sb - sc;
    corners[6] = center - sa - sb + sc;
    corners[7] = center - sa - sb - sc;

    for (int i = 0; i < 8; ++i) {
        bounds.lower.x = std::min(bounds.lower.x, corners[i].x);
        bounds.lower.y = std::min(bounds.lower.y, corners[i].y);
        bounds.lower.z = std::min(bounds.lower.z, corners[i].z);

        bounds.upper.x = std::max(bounds.upper.x, corners[i].x);
        bounds.upper.y = std::max(bounds.upper.y, corners[i].y);
        bounds.upper.z = std::max(bounds.upper.z, corners[i].z);
    }
  }


  void Model::assign_tensor_pointers(
                              float *aabbs,
                              int numEllipsoids) {
    // 1. Free existing memory
    if (this->aabbs) cudaFree(this->aabbs);
    
    // 2. Allocate new memory
    cudaMalloc((void**)&this->aabbs, numEllipsoids * 6 * sizeof(float));

    // 3. Copy data to device
    cudaMemcpy(this->aabbs, aabbs, numEllipsoids * 6 * sizeof(float), cudaMemcpyHostToDevice);
    
    this->numEllipsoids = numEllipsoids;
  }

  void Model::updateSize(int newSize) {
    this->numEllipsoids = newSize;
  }

} // ::osc
