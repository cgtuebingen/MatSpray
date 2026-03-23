struct Camera {
  float3 position;
  float3 direction;
  float3 horizontal;
  float3 vertical;
};

struct PerRayData : IDifferentiable {
  float transmittance;
  bool noHit;
  float tMin;
  uint32_t intGaussianId;
};

// LaunchParams
RWStructuredBuffer<float> centers;
RWStructuredBuffer<float> scales;
RWStructuredBuffer<float> rotations;
RWStructuredBuffer<float> densities;

RWStructuredBuffer<uint32_t> intGaussianIdsBuffer;
RWStructuredBuffer<uint32_t> numIntersectionsBuffer;
RWStructuredBuffer<uint32_t> prev_indexBuffer;
RWStructuredBuffer<float> depthBuffer;

int2 fbSize;
Camera camera;
float epsilon;
uint32_t max_num_intersections;
float transmittance_threshold;
uint32_t numEllipsoids;
float depth_max_tolerance;
RaytracingAccelerationStructure traversable;


static const int SUPERSAMPLING_MODE = 1; // 0=disabled, 1=regular grid, 2=jittered
static const int SAMPLES_PER_PIXEL_X = 8;
static const int SAMPLES_PER_PIXEL_Y = 8;
static const float SHIFT_X = 0.f;
static const float SHIFT_Y = 0.f;

float random(uint seed) {
    uint x = seed;
    x ^= x >> 13;
    x *= 0x9e3779b9;
    x ^= x >> 16;
    return float(x) / 4294967296.0f;
}

[Differentiable]
float3 matmul(const float mat[3][3], const float3 vec) {
        return float3(mat[0][0] * vec.x + mat[0][1] * vec.y + mat[0][2] * vec.z,
                     mat[1][0] * vec.x + mat[1][1] * vec.y + mat[1][2] * vec.z,
                     mat[2][0] * vec.x + mat[2][1] * vec.y + mat[2][2] * vec.z);
}

[Differentiable]
float compute_gaussian_opacity_at_point(float3 world_point, float3 center, float3 scale, float invRotation[3][3], float base_opacity, float beta) {
    float3 oc = world_point - center;
    float3 local_point = matmul(invRotation, oc);
    
    float invScaleMat[3][3] = {
        { 1.f / scale.x, 0.f, 0.f },
        { 0.f, 1.f / scale.y, 0.f },
        { 0.f, 0.f, 1.f / scale.z }
    };
    
    local_point = matmul(invScaleMat, local_point);
    float squared_distance = dot(local_point, local_point);
    float gaussian_falloff = exp(-beta * squared_distance);
    
    return base_opacity * gaussian_falloff;
}

[Differentiable]
float3 find_point_of_maximum_response(float3 ray_origin, float3 ray_direction, float3 gaussian_center, float t_min, float t_max) {
    float3 oc = gaussian_center - ray_origin;
    float t_closest = dot(oc, ray_direction) / dot(ray_direction, ray_direction);
    t_closest = max(t_min, min(t_max, t_closest));
    return ray_origin + t_closest * ray_direction;
}

[shader("intersection")]
void ellipsoid_intersection()
{
    const int ix = DispatchRaysIndex().x;
    const int iy = DispatchRaysIndex().y;
    const uint fbIndex = ix + iy * fbSize.x;

    float3 rayOrigin = ObjectRayOrigin();
    float3 rayDirection = ObjectRayDirection();
    const int primID = PrimitiveIndex();

    if (primID == -1) {
        printf("The primitive id was -1\n");
        return;
    }

    float3 center = float3(centers[primID * 3 + 0],
                            centers[primID * 3 + 1],
                            centers[primID * 3 + 2]);

    float3 scale = float3(scales[primID * 3 + 0],
                          scales[primID * 3 + 1],
                          scales[primID * 3 + 2]);

    float invRotation[3][3] = {
        { rotations[primID * 9 + 0], rotations[primID * 9 + 3], rotations[primID * 9 + 6] },
        { rotations[primID * 9 + 1], rotations[primID * 9 + 4], rotations[primID * 9 + 7] },
        { rotations[primID * 9 + 2], rotations[primID * 9 + 5], rotations[primID * 9 + 8] }
    };

    float3 oc = rayOrigin - center;
    float3 localRayOrigin = matmul(invRotation, oc);
    float3 localRayDirection = matmul(invRotation, rayDirection);

    scale = max(scale, 1e-6);

    float invScaleMat[3][3] = {
        { 1.f / scale.x, 0.f, 0.f },
        { 0.f, 1.f / scale.y, 0.f },
        { 0.f, 0.f, 1.f / scale.z }
    };

    localRayOrigin = matmul(invScaleMat, localRayOrigin);
    localRayDirection = matmul(invScaleMat, localRayDirection);

    float a = dot(localRayDirection, localRayDirection);
    float a_sign = a < 0.f ? -1.f : 1.f;
    a = max(a, a_sign * epsilon);
    float b_prime = dot(-localRayOrigin, localRayDirection);

    float3 delta_brackets = localRayOrigin + (b_prime / a) * localRayDirection;
    float delta = 1.f - dot(delta_brackets, delta_brackets);

    if (delta > 0.0f)
    {
        float c = dot(localRayOrigin, localRayOrigin) - 1.f;

        if (b_prime < 0.f && c > 0.f) {
          return;
        }

        float b_prime_sign = b_prime < 0.f ? -1.f : 1.f;
        float q = b_prime + b_prime_sign * sqrt(a*delta);
        float q_sign = q < 0.f ? -1.f : 1.f;
        q = max(q, q_sign * epsilon);
        float t0 = c / q;
        float t1 = q / a;

        if (t0 > 0.0f && t1 > 0.0f && t0 > RayTMin()) {
            if (prev_indexBuffer[fbIndex] == primID) {
                return;
            }

            if (t0 >= t1) {
              return;
            }

            float3 ray_origin_world = WorldRayOrigin();
            float3 ray_dir_world = WorldRayDirection();
            float3 max_response_point = find_point_of_maximum_response(ray_origin_world, ray_dir_world, center, t0, t1);
            float base_opacity = densities[primID];
            float gaussian_opacity = compute_gaussian_opacity_at_point(max_response_point, center, scale, invRotation, base_opacity, 2.0f);
            
            if (gaussian_opacity >= 0.005f) {
                ReportHitOptix(t0, 0, asuint(t1));
            }
        }
    }
}

[shader("closesthit")]
void closesthit_radiance(
  inout PerRayData prd : SV_RayPayload,
  uint32_t out_dist : SV_Attribute0)
{
  const int ix = DispatchRaysIndex().x;
  const int iy = DispatchRaysIndex().y;
  const uint fbIndex = ix + iy * fbSize.x;
  const int primIdx = PrimitiveIndex();
  prev_indexBuffer[fbIndex] = primIdx;

  const float3 rayOrigin = WorldRayOrigin();
  const float3 rayDir = WorldRayDirection();
  const float t0 = RayTCurrent();
  const float t1 = asfloat(out_dist);

  float3 center = float3(centers[primIdx * 3 + 0],
                         centers[primIdx * 3 + 1],
                         centers[primIdx * 3 + 2]);

  float3 scale = float3(scales[primIdx * 3 + 0],
                        scales[primIdx * 3 + 1],
                        scales[primIdx * 3 + 2]);
  
  scale = max(scale, 1e-6);

  float invRotation[3][3] = {
      { rotations[primIdx * 9 + 0], rotations[primIdx * 9 + 3], rotations[primIdx * 9 + 6] },
      { rotations[primIdx * 9 + 1], rotations[primIdx * 9 + 4], rotations[primIdx * 9 + 7] },
      { rotations[primIdx * 9 + 2], rotations[primIdx * 9 + 5], rotations[primIdx * 9 + 8] }
  };

  float3 max_response_point = find_point_of_maximum_response(rayOrigin, rayDir, center, t0, t1);
  float base_opacity = densities[primIdx];
  float gaussian_opacity = compute_gaussian_opacity_at_point(max_response_point, center, scale, invRotation, base_opacity, 0.5f);

  prd.tMin = RayTCurrent() + epsilon;
  prd.intGaussianId = primIdx;
  prd.transmittance *= (1.0f - gaussian_opacity);
}

[shader("anyhit")]
void anyhit_radiance() {
}

[shader("miss")]
void miss_radiance(inout PerRayData prd : SV_RayPayload)
{
  prd.noHit = true;
}

[shader("raygeneration")]
void renderFrame()
{
  const int ix = DispatchRaysIndex().x;
  const int iy = DispatchRaysIndex().y;

  const uint fbIndex = ix + iy * fbSize.x;

  const float min_transmittance = 0.0001f;
  float maxDepth = depthBuffer[fbIndex];
  bool useDepthGate = maxDepth > 0.f;

  if (SUPERSAMPLING_MODE == 0) {
    PerRayData rayData;
    rayData.transmittance = 1.f;
    rayData.noHit = false;
    rayData.tMin = epsilon;

    const float2 screen = float2(ix + .5f, iy + .5f) * float2(1.f / fbSize.x, 1.f / fbSize.y);

    RayDesc ray;
    ray.Origin = camera.position;
    ray.Direction = normalize(camera.direction
                              + (screen.x - 0.5f + SHIFT_X) * camera.horizontal
                              + (screen.y - 0.5f + SHIFT_Y) * camera.vertical);
    ray.TMin = epsilon;
    ray.TMax = 1e20f;

    prev_indexBuffer[fbIndex] = -1;
    uint32_t loop_index = 0;

    while(!rayData.noHit && loop_index < 128 && rayData.transmittance > min_transmittance && (!useDepthGate || ray.TMin <= maxDepth + depth_max_tolerance)) {
        TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, rayData);
        ray.TMin = rayData.tMin;

        if (!rayData.noHit) {
            loop_index++;
            if (loop_index < 150) {
                intGaussianIdsBuffer[fbIndex * 150 + loop_index] = rayData.intGaussianId;
            }
        }
    }
    numIntersectionsBuffer[fbIndex] = loop_index;
    return;
  }

  // Super sampling implementation
  const int total_samples = SAMPLES_PER_PIXEL_X * SAMPLES_PER_PIXEL_Y;
  const float inv_samples_x = 1.0f / float(SAMPLES_PER_PIXEL_X);
  const float inv_samples_y = 1.0f / float(SAMPLES_PER_PIXEL_Y);
  
  uint32_t combined_loop_index = 0;
  uint32_t max_intersections_per_sample = 150 / total_samples;
  
  for (int i = 0; i < 150; i++) {
    intGaussianIdsBuffer[fbIndex * 150 + i] = uint32_t(-1);
  }

  for (int sy = 0; sy < SAMPLES_PER_PIXEL_Y; sy++) {
    for (int sx = 0; sx < SAMPLES_PER_PIXEL_X; sx++) {
      PerRayData rayData;
      rayData.transmittance = 1.f;
      rayData.noHit = false;
      rayData.tMin = epsilon;

      float2 offset;
      if (SUPERSAMPLING_MODE == 1) {
        offset = float2((sx + 0.5f) * inv_samples_x, (sy + 0.5f) * inv_samples_y);
      } else {
        uint seed = fbIndex * 1000 + sy * SAMPLES_PER_PIXEL_X + sx;
        offset = float2((sx + random(seed)) * inv_samples_x, 
                       (sy + random(seed + 1)) * inv_samples_y);
      }

      const float2 screen = float2(ix + offset.x, iy + offset.y) * float2(1.f / fbSize.x, 1.f / fbSize.y);

      RayDesc ray;
      ray.Origin = camera.position;
      ray.Direction = normalize(camera.direction
                                + (screen.x - 0.5f + SHIFT_X) * camera.horizontal
                                + (screen.y - 0.5f + SHIFT_Y) * camera.vertical);
      ray.TMin = epsilon;
      ray.TMax = 1e20f;

      prev_indexBuffer[fbIndex] = -1;
      uint32_t loop_index = 0;

      while(!rayData.noHit && loop_index < max_intersections_per_sample && rayData.transmittance > min_transmittance) {
          TraceRay(traversable, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, rayData);
          ray.TMin = rayData.tMin;

        if (!rayData.noHit) {
            bool already_intersected = false;
            for (uint32_t check_i = 0; check_i < combined_loop_index; check_i++) {
              if (intGaussianIdsBuffer[fbIndex * 150 + check_i] == rayData.intGaussianId) {
                  already_intersected = true;
                  break;
              }
            }
            
            if (combined_loop_index < 150 && !already_intersected) {
                intGaussianIdsBuffer[fbIndex * 150 + combined_loop_index] = rayData.intGaussianId;
                combined_loop_index++;
                loop_index++;
            }
        }
      }
    }
  }

  numIntersectionsBuffer[fbIndex] = combined_loop_index;
}
