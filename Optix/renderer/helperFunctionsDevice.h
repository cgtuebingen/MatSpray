//
// Created by langstei on 10/30/24.
//

#include <cstdint>

namespace osc {
    extern "C" void convertQuaternionToRotationMatrix(float *quaternionsDevice, float *rotationMatricesDevice, int numEllipsoids);
    extern "C" void calculateBoundingBoxes(float *centers, float *scales, float *rotations, float *boundingBoxes, int numEllipsoids);
    extern "C" void calc_rot_gradients(float *quaternions, float *normalized_quaternions, float *grad_rotations, float *grad_rotations_mat, int numEllipsoids);
    extern "C" void processIntersectionColors(uint32_t *intersection_ids, uint32_t *num_intersections,
                                             float *pixel_colors, float *gaussian_colors, 
                                             int32_t *gaussian_hit_counts,
                                             int height, int width, int max_intersections, int num_gaussians);
    extern "C" void processIntersectionMedianColors(uint32_t *intersection_ids, uint32_t *num_intersections,
                                                   float *pixel_colors, float *gaussian_color_values,
                                                   float *gaussian_median_colors, int32_t *gaussian_hit_counts,
                                                   int height, int width, int max_intersections, 
                                                   int num_gaussians, int max_hits_per_gaussian);
};
