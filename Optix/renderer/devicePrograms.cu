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

#include <cfloat>
#include <optix_device.h>

#include "LaunchParams.h"
#include <curand_kernel.h>

using namespace osc;

namespace osc {
    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */

    // for this simple example, we have a single ray type
    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
    void *unpackPointer(uint32_t i0, uint32_t i1) {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void *ptr = reinterpret_cast<void *>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
    void packPointer(void *ptr, uint32_t &i0, uint32_t &i1) {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRDColor() {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>(unpackPointer(u0, u1));
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRDHitpoint() {
        const uint32_t u0 = optixGetPayload_2();
        const uint32_t u1 = optixGetPayload_3();
        return reinterpret_cast<T *>(unpackPointer(u0, u1));
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRDTransmittance() {
        const uint32_t u0 = optixGetPayload_4();
        const uint32_t u1 = optixGetPayload_5();
        return reinterpret_cast<T *>(unpackPointer(u0, u1));
    }

    // Function to calculate the determinant of a 3x3 matrix
    __device__ float determinant(const float mat[3][3]) {
        return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
               - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
               + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    }

    // Function to calculate the inverse of a 3x3 matrix
    __device__ void inverse(const float mat[3][3], float inv[3][3]) {
        float det = determinant(mat);

        float invDet = 1.0f / det;

        inv[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * invDet;
        inv[0][1] = -(mat[0][1] * mat[2][2] - mat[0][2] * mat[2][1]) * invDet;
        inv[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * invDet;

        inv[1][0] = -(mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) * invDet;
        inv[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * invDet;
        inv[1][2] = -(mat[0][0] * mat[1][2] - mat[0][2] * mat[1][0]) * invDet;

        inv[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * invDet;
        inv[2][1] = -(mat[0][0] * mat[2][1] - mat[0][1] * mat[2][0]) * invDet;
        inv[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * invDet;
    }

    __device__ vec3f matmul(const float mat[3][3], const vec3f vec) {
        return vec3f(mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z,
                     mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z,
                     mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z);
    }

    __device__ void transpose3x3(float input[3][3], float output[3][3]) {
        output[0][1] = input[1][0];
        output[0][2] = input[2][0];
        output[1][0] = input[0][1];
        output[1][2] = input[2][1];
        output[2][0] = input[0][2];
        output[2][1] = input[1][2];
        output[0][0] = input[0][0];
        output[1][1] = input[1][1];
        output[2][2] = input[2][2];
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __intersection__ellipsoid() {
        const float3 rayOrigin = optixGetObjectRayOrigin();
        const float3 rayDirection = optixGetObjectRayDirection();

        // Retrieve ellipsoid data from SBT (center, scale, and rotation)
        const GaussianSBTData &sbtData = *(const GaussianSBTData *) optixGetSbtDataPointer();
        const int primID = optixGetPrimitiveIndex();

        // Access center and scale directly using pointer arithmetic
        float3 center = make_float3(*(sbtData.centers.data + primID * 3 + 0),
                                    *(sbtData.centers.data + primID * 3 + 1),
                                    *(sbtData.centers.data + primID * 3 + 2));

        float3 scale = make_float3(*(sbtData.scales.data + primID * 3 + 0),
                                *(sbtData.scales.data + primID * 3 + 1),
                                *(sbtData.scales.data + primID * 3 + 2));

        // Load rotation matrix as a 3x3 matrix from SBT data (stored as a flat array of floats)
        float invRotation[3][3] = {
            {*(sbtData.rotations.data + primID * 9 + 0), *(sbtData.rotations.data + primID * 9 + 3), *(sbtData.rotations.data + primID * 9 + 6)},
            {*(sbtData.rotations.data + primID * 9 + 1), *(sbtData.rotations.data + primID * 9 + 4), *(sbtData.rotations.data + primID * 9 + 7)},
            {*(sbtData.rotations.data + primID * 9 + 2), *(sbtData.rotations.data + primID * 9 + 5), *(sbtData.rotations.data + primID * 9 + 8)}
        };

        // Translate the ray origin to ellipsoid's local space by subtracting the center
        float3 oc = make_float3(rayOrigin.x - center.x, rayOrigin.y - center.y, rayOrigin.z - center.z);

        // Apply the inverse of the ellipsoid's rotation to both the ray origin (oc) and direction
        // This brings the ray into the ellipsoid's local space where it's axis-aligned
        float3 localRayOrigin = matmul(invRotation, oc);
        float3 localRayDirection = matmul(invRotation, rayDirection);

        // Apply inverse scale to bring the ray into the unit sphere's space
        float invScaleMat[3][3] = {{1.f / scale.x, 0.f, 0.f}, {0.f, 1.f / scale.y, 0.f}, {0.f, 0.f, 1.f / scale.z}};

        localRayOrigin = matmul(invScaleMat, localRayOrigin);
        localRayDirection = matmul(invScaleMat, localRayDirection);

        // Compute coefficients for the quadratic equation
        float a = localRayDirection.x * localRayDirection.x
                + localRayDirection.y * localRayDirection.y
                + localRayDirection.z * localRayDirection.z;

        float b_prime = -localRayOrigin.x * localRayDirection.x
                        - localRayOrigin.y * localRayDirection.y
                        - localRayOrigin.z * localRayDirection.z;

        float3 delta_brackets = make_float3(localRayOrigin.x + (b_prime / a) * localRayDirection.x,
                                            localRayOrigin.y + (b_prime / a) * localRayDirection.y,
                                            localRayOrigin.z + (b_prime / a) * localRayDirection.z);

        float delta = 1.f - (delta_brackets.x * delta_brackets.x
                            + delta_brackets.y * delta_brackets.y
                            + delta_brackets.z * delta_brackets.z);

        // Check if the ray intersects the ellipsoid
        if (delta >= 0.0f) {
            float c = localRayOrigin.x * localRayOrigin.x + localRayOrigin.y * localRayOrigin.y + localRayOrigin.z * localRayOrigin.z - 1.f;
            float b_prime_sign = b_prime < 0.f ? -1.f : 1.f;
            float q = b_prime + b_prime_sign * sqrt(a * delta);
            float t0 = c / q;
            float t1 = q / a;

            // Find the nearest valid intersection point
            if (t0 > 0.0f && t1 > 0.0f) {
                float t = fminf(t0, t1);
                optixReportIntersection(t + 1e-6f, 0);
            }
        }
    }



    // extern "C" __global__ void __closesthit__radiance() {
    //     const GaussianSBTData &sbtData = *(const GaussianSBTData *) optixGetSbtDataPointer();
    //     const int primID = optixGetPrimitiveIndex();
    //     const float t = optixGetRayTmax();
    //     const vec3f oldHitpoint = optixGetWorldRayOrigin();
    //
    //     Ellipsoid ellipsoid = *(sbtData.ellipsoids.data + primID);
    //
    //     PerRayData &rayData = *(PerRayData *) getPRDColor<PerRayData>();
    //
    //     const vec3f rayDir = optixGetWorldRayDirection();
    //
    //     vec3f color;
    //     eval_shs(rayDir, ellipsoid.sh, color, sbtData.shDegree);
    //
    //     rayData.color += color * ellipsoid.opacity * rayData.transmittance;
    //     rayData.transmittance *= (1.f - ellipsoid.opacity);
    //
    //     rayData.hitpoint = oldHitpoint + t * rayDir;
    // }

    extern "C" __global__ void __anyhit__radiance() {
        // printf("__anyhit__radiance\n");
        //
        // vec3f &prdColor = *(vec3f*) getPRDColor<vec3f>();
        // vec3f &prdHitPoint = *(vec3f*) getPRDHitpoint<vec3f>();
        // //float &prdTransmittance = *(float*) getPRDTransmittance<float>();
        // // set to constant white as background color
        // prdColor = vec3f(0.1f);
        // //prdTransmittance = -1.f;
    }


    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance() {
        PerRayData &rayData = *(PerRayData *) getPRDColor<PerRayData>();
        // set to constant white as background color
        // rayData.color += vec3f(1.f) * rayData.transmittance; // Commented out - no color field in PerRayData
        rayData.noHit = true;
    }

    __device__ float halton(int index, int base) {
        float result = 0.0f;
        float f = 1.0f / base;
        int i = index;
        while (i > 0) {
            result = result + f * (i % base);
            i = i / base;
            f = f / base;
        }
        return result;
    }


    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    // extern "C" __global__ void __raygen__renderFrame() {
    //     // Get the launch index for the current pixel
    //     const int ix = optixGetLaunchIndex().x;
    //     const int iy = optixGetLaunchIndex().y;
    //
    //     const auto &camera = optixLaunchParams.camera;
    //     constexpr int max_int_per_ray = 300;
    //
    //     // Initialize final pixel color
    //     vec3f finalColor = vec3f(0.f);
    //
    //     // Number of samples per pixel
    //     const int numSamples = 2;
    //
    //     // Loop over the number of samples
    //     for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx) {
    //         // Halton sequence for the current sample (base 2 for x, base 3 for y)
    //         float intersections[max_int_per_ray];
    //         float haltonX = halton(sampleIdx, 2);
    //         float haltonY = halton(sampleIdx, 3);
    //
    //         // Offset normalized screen plane position with Halton jitter
    //         vec2f screen(vec2f(ix + haltonX, iy + haltonY)
    //                      / vec2f(optixLaunchParams.frame.size));
    //
    //         // Generate ray direction with jittered screen coordinates
    //         vec3f rayDir = normalize(camera.direction
    //                                  + (screen.x - 0.5f) * camera.horizontal
    //                                  + (screen.y - 0.5f) * camera.vertical);
    //
    //         PerRayData rayData;
    //
    //         rayData.color = vec3f(0.f);
    //         rayData.hitpoint = camera.position;
    //         rayData.transmittance = 1.f;
    //         rayData.curDensity = 0.0f;
    //         rayData.curNumOfVolumes = 0;
    //         rayData.noHit = false;
    //
    //         uint32_t u0, u1;
    //
    //         packPointer(&rayData, u0, u1);
    //
    //         int counter = 0;
    //
    //         while (rayData.transmittance >= 0.0001f && !rayData.noHit && counter < max_int_per_ray) {
    //             optixTrace(optixLaunchParams.traversable,
    //                        rayData.hitpoint,
    //                        rayDir,
    //                        0.f, // tmin
    //                        1e20f, // tmax
    //                        0.0f, // rayTime
    //                        OptixVisibilityMask(255),
    //                        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    //                        SURFACE_RAY_TYPE, // SBT offset
    //                        RAY_TYPE_COUNT, // SBT stride
    //                        SURFACE_RAY_TYPE, // missSBTIndex
    //                        u0, u1);
    //             counter++;
    //         }
    //
    //         // Accumulate the result from this sample
    //         finalColor += rayData.color;
    //     }
    //
    //     finalColor /= numSamples;
    //
    //
    //     // Convert the color to 8-bit per channel and pack it into an RGBA format
    //     const int r = int(255.f * min(finalColor.x, 1.f));
    //     const int g = int(255.f * min(finalColor.y, 1.f));
    //     const int b = int(255.f * min(finalColor.z, 1.f));
    //
    //     // Convert to 32-bit rgba value
    //     const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
    //
    //     // Write the final color to the frame buffer
    //     const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    //     //optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    //     *(optixLaunchParams.frame.colorBuffer.data + fbIndex) = rgba;
    // }
} // ::osc
