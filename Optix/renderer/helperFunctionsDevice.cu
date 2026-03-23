#include <curand_mtgp32_kernel.h>
#include <cfloat>

#include "LaunchParams.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "helperFunctionsDevice.h"

using namespace osc;

namespace osc {
    __device__ void quaternion_to_rotation_matrix(const float *q, float *rotation_matrix) {
        float w = q[0];
        float x = q[1];
        float y = q[2];
        float z = q[3];

        rotation_matrix[0] = 1 - 2 * (y * y + z * z);
        rotation_matrix[1] = 2 * (x * y - w * z);
        rotation_matrix[2] = 2 * (x * z + w * y);

        rotation_matrix[3] = 2 * (x * y + w * z);
        rotation_matrix[4] = 1 - 2 * (x * x + z * z);
        rotation_matrix[5] = 2 * (y * z - w * x);

        rotation_matrix[6] = 2 * (x * z - w * y);
        rotation_matrix[7] = 2 * (y * z + w * x);
        rotation_matrix[8] = 1 - 2 * (x * x + y * y);
    }

    __device__ void calc_quaternion_gradient(const float *quaternion, float *grad_rotation, float *grad_rotation_mat){
        const float g_m0 = grad_rotation_mat[0];
        const float g_m1 = grad_rotation_mat[1];
        const float g_m2 = grad_rotation_mat[2];
        const float g_m3 = grad_rotation_mat[3];
        const float g_m4 = grad_rotation_mat[4];
        const float g_m5 = grad_rotation_mat[5];
        const float g_m6 = grad_rotation_mat[6];
        const float g_m7 = grad_rotation_mat[7];
        const float g_m8 = grad_rotation_mat[8];

        const float w = quaternion[0];
        const float x = quaternion[1];
        const float y = quaternion[2];
        const float z = quaternion[3];

        grad_rotation[0] = 2 * x * (g_m7 - g_m5) + 2 * y * (g_m2 - g_m6) + 2 * z * (g_m3 - g_m1);
        grad_rotation[1] = 2 * w * (g_m7 - g_m5) + 2 * y * (g_m1 + g_m3) + 2 * z * (g_m2 + g_m6) - 4 * x * (g_m4 + g_m8);
        grad_rotation[2] = 2 * w * (g_m2 - g_m6) + 2 * x * (g_m1 + g_m3) + 2 * z * (g_m5 + g_m7) - 4 * y * (g_m0 + g_m8);
        grad_rotation[3] = 2 * w * (g_m3 - g_m1) + 2 * x * (g_m2 + g_m6) + 2 * y * (g_m5 + g_m7) - 4 * z * (g_m0 + g_m4);
    }

    __device__ void calc_qauternion_norm_gradient(const float *quaternion, float *grad_rotation){
        const float w = quaternion[0];
        const float x = quaternion[1];
        const float y = quaternion[2];
        const float z = quaternion[3];

        float denom = w * w + x * x + y * y + z * z;
        float denom_sq = denom * denom;
        denom_sq = max(denom_sq, 1e-6);

        grad_rotation[0] *= (-w*w + x*x + y*y + z*z)/denom_sq;
        grad_rotation[1] *= (w*w - x*x + y*y + z*z)/denom_sq;
        grad_rotation[2] *= (w*w + x*x - y*y + z*z)/denom_sq;
        grad_rotation[3] *= (w*w + x*x + y*y - z*z)/denom_sq;
    }

    __global__ void calculateRotationMatrices(
        const float *quaternionsDevice, // Flattened array of quaternions
        float *rotationMatricesDevice, // Output buffer for rotation matrices
        int numEllipsoids // Number of ellipsoids
    ) {
        unsigned int ellipsoidID = blockIdx.x * blockDim.x + threadIdx.x;
        if (ellipsoidID >= numEllipsoids) return;

        // Pointer to the current quaternion in the flattened array
        const float *quaternion = &quaternionsDevice[ellipsoidID * 4];

        // Pointer to the current 3x3 rotation matrix in the output buffer
        float *rotation_matrix = &rotationMatricesDevice[ellipsoidID * 9];

        // Convert quaternion to rotation matrix
        quaternion_to_rotation_matrix(quaternion, rotation_matrix);
    }

    __global__ void calc_rot_gradients_helper(
        const float *quaternions, 
        const float *normalized_quaternions,
        float *grad_rotations, 
        float *grad_rotations_mats,
        int numEllipsoids){
        unsigned int ellipsoidID = blockIdx.x * blockDim.x + threadIdx.x;
        if (ellipsoidID >= numEllipsoids) return;

        const float *quaternion = &quaternions[ellipsoidID * 4];

        const float *normalized_quaternion = &normalized_quaternions[ellipsoidID * 4];

        float *grad_rotation = &grad_rotations[ellipsoidID * 4];

        float *grad_rotation_mat = &grad_rotations_mats[ellipsoidID * 9];

        calc_quaternion_gradient(normalized_quaternion, grad_rotation, grad_rotation_mat);

        calc_qauternion_norm_gradient(quaternion, grad_rotation);
    }

    __device__ void calcAABB(const float *center, const float *scale, const float *rotation_matrix,
                             float *bounding_box) {
        // Initialize the bounding box extremes
        float min_x = FLT_MAX;
        float min_y = FLT_MAX;
        float min_z = FLT_MAX;
        float max_x = -FLT_MAX;
        float max_y = -FLT_MAX;
        float max_z = -FLT_MAX;

        // Calculate the scaled rotation vectors
        float sa[3] = {scale[0] * rotation_matrix[0], scale[0] * rotation_matrix[3], scale[0] * rotation_matrix[6]};
        float sb[3] = {scale[1] * rotation_matrix[1], scale[1] * rotation_matrix[4], scale[1] * rotation_matrix[7]};
        float sc[3] = {scale[2] * rotation_matrix[2], scale[2] * rotation_matrix[5], scale[2] * rotation_matrix[8]};

        // Define corners based on combinations of sa, sb, and sc
        float corners[8][3] = {
            {center[0] + sa[0] + sb[0] + sc[0], center[1] + sa[1] + sb[1] + sc[1], center[2] + sa[2] + sb[2] + sc[2]},
            {center[0] + sa[0] + sb[0] - sc[0], center[1] + sa[1] + sb[1] - sc[1], center[2] + sa[2] + sb[2] - sc[2]},
            {center[0] + sa[0] - sb[0] + sc[0], center[1] + sa[1] - sb[1] + sc[1], center[2] + sa[2] - sb[2] + sc[2]},
            {center[0] + sa[0] - sb[0] - sc[0], center[1] + sa[1] - sb[1] - sc[1], center[2] + sa[2] - sb[2] - sc[2]},
            {center[0] - sa[0] + sb[0] + sc[0], center[1] - sa[1] + sb[1] + sc[1], center[2] - sa[2] + sb[2] + sc[2]},
            {center[0] - sa[0] + sb[0] - sc[0], center[1] - sa[1] + sb[1] - sc[1], center[2] - sa[2] + sb[2] - sc[2]},
            {center[0] - sa[0] - sb[0] + sc[0], center[1] - sa[1] - sb[1] + sc[1], center[2] - sa[2] - sb[2] + sc[2]},
            {center[0] - sa[0] - sb[0] - sc[0], center[1] - sa[1] - sb[1] - sc[1], center[2] - sa[2] - sb[2] - sc[2]}
        };

        // Calculate the bounding box by iterating over all corners
        for (int i = 0; i < 8; i++) {
            min_x = fminf(min_x, corners[i][0]);
            min_y = fminf(min_y, corners[i][1]);
            min_z = fminf(min_z, corners[i][2]);
            max_x = fmaxf(max_x, corners[i][0]);
            max_y = fmaxf(max_y, corners[i][1]);
            max_z = fmaxf(max_z, corners[i][2]);
        }

        // Write results to the bounding box array
        bounding_box[0] = min_x;
        bounding_box[1] = min_y;
        bounding_box[2] = min_z;
        bounding_box[3] = max_x;
        bounding_box[4] = max_y;
        bounding_box[5] = max_z;
    }

    __global__ void calculateAABBs(
        float *centers, // Flattened array of ellipsoid centers
        float *scales, // Flattened array of ellipsoid scales
        float *rotations, // Flattened array of ellipsoid rotations
        float *boundingBoxes, // Output buffer for bounding boxes
        int numEllipsoids // Number of ellipsoids
    ) {
        unsigned int ellipsoidID = blockIdx.x * blockDim.x + threadIdx.x;
        if (ellipsoidID >= numEllipsoids) return;

        // Pointer to the current center in the flattened array
        const float *center = &centers[ellipsoidID * 3];

        // Pointer to the current scale in the flattened array
        const float *scale = &scales[ellipsoidID * 3];

        // Pointer to the current rotation matrix in the flattened array
        const float *rotation_matrix = &rotations[ellipsoidID * 9];

        // Pointer to the current bounding box in the output buffer
        float *bounding_box = &boundingBoxes[ellipsoidID * 6];

        calcAABB(center, scale, rotation_matrix, bounding_box);
    }

    __device__ float findMedian(float* values, int count) {
        // Simple bubble sort for small arrays (could be optimized for larger datasets)
        for (int i = 0; i < count - 1; i++) {
            for (int j = 0; j < count - i - 1; j++) {
                if (values[j] > values[j + 1]) {
                    float temp = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = temp;
                }
            }
        }
        
        if (count % 2 == 0) {
            // Even number of elements - return average of middle two
            return (values[count/2 - 1] + values[count/2]) / 2.0f;
        } else {
            // Odd number of elements - return middle element
            return values[count/2];
        }
    }

    __global__ void processIntersectionColorsForMedianKernel(
        const uint32_t *intersection_ids,     // [height, width, max_intersections] - Gaussian IDs per pixel
        const uint32_t *num_intersections,    // [height, width] - number of intersections per pixel
        const float *pixel_colors,            // [height, width, 3] - RGB color values per pixel
        float *gaussian_color_values,         // [num_gaussians, max_hits_per_gaussian, 3] - all color values per Gaussian
        int32_t *gaussian_hit_counts,         // [num_gaussians] - output: hit count per Gaussian
        int height,                           // Image height
        int width,                            // Image width
        int max_intersections,                // Maximum intersections per pixel
        int num_gaussians,                    // Total number of Gaussians
        int max_hits_per_gaussian             // Maximum hits we can store per Gaussian
    ) {
        // Calculate pixel coordinates
        int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = height * width;
        
        if (pixel_id >= total_pixels) return;
        
        // Get number of intersections for this pixel
        int num_hits = num_intersections[pixel_id];
        
        if (num_hits > 0) {
            // Get pixel color [R, G, B]
            const float *pixel_color = &pixel_colors[pixel_id * 3];
            
            // Process each intersection at this pixel
            for (int i = 0; i < num_hits && i < max_intersections; i++) {
                uint32_t gaussian_id = intersection_ids[pixel_id * max_intersections + i];
                
                if (gaussian_id < num_gaussians) {
                    // Atomically increment hit count and get the index where we should store this color
                    int hit_index = atomicAdd(&gaussian_hit_counts[gaussian_id], 1);
                    
                    if (hit_index < max_hits_per_gaussian) {
                        // Store the color values at the appropriate index
                        int base_idx = gaussian_id * max_hits_per_gaussian * 3 + hit_index * 3;
                        gaussian_color_values[base_idx + 0] = pixel_color[0];  // Red
                        gaussian_color_values[base_idx + 1] = pixel_color[1];  // Green
                        gaussian_color_values[base_idx + 2] = pixel_color[2];  // Blue
                    }
                }
            }
        }
    }

    __global__ void calculateMedianGaussianColors(
        const float *gaussian_color_values,   // [num_gaussians, max_hits_per_gaussian, 3] - all color values
        const int32_t *gaussian_hit_counts,   // [num_gaussians] - hit counts
        float *gaussian_median_colors,        // [num_gaussians, 3] - output: median colors per Gaussian
        int num_gaussians,                    // Total number of Gaussians
        int max_hits_per_gaussian             // Maximum hits per Gaussian
    ) {
        int gaussian_id = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (gaussian_id >= num_gaussians) return;
        
        int32_t hit_count = gaussian_hit_counts[gaussian_id];
        
        if (hit_count > 0) {
            // Limit hit_count to max_hits_per_gaussian in case we had overflow
            hit_count = min(hit_count, max_hits_per_gaussian);
            
            // Temporary arrays to hold values for each channel
            float red_values[1024];   // Adjust size based on expected max hits
            float green_values[1024];
            float blue_values[1024];
            
            // Copy values to temporary arrays
            for (int i = 0; i < hit_count; i++) {
                int base_idx = gaussian_id * max_hits_per_gaussian * 3 + i * 3;
                red_values[i] = gaussian_color_values[base_idx + 0];
                green_values[i] = gaussian_color_values[base_idx + 1];
                blue_values[i] = gaussian_color_values[base_idx + 2];
            }
            
            // Calculate median for each channel
            gaussian_median_colors[gaussian_id * 3 + 0] = findMedian(red_values, hit_count);   // Red
            gaussian_median_colors[gaussian_id * 3 + 1] = findMedian(green_values, hit_count); // Green
            gaussian_median_colors[gaussian_id * 3 + 2] = findMedian(blue_values, hit_count);  // Blue
        } else {
            // No hits - set to zero
            gaussian_median_colors[gaussian_id * 3 + 0] = 0.0f;
            gaussian_median_colors[gaussian_id * 3 + 1] = 0.0f;
            gaussian_median_colors[gaussian_id * 3 + 2] = 0.0f;
        }
    }

    __global__ void processIntersectionColorsKernel(
        const uint32_t *intersection_ids,     // [height, width, max_intersections] - Gaussian IDs per pixel
        const uint32_t *num_intersections,    // [height, width] - number of intersections per pixel
        const float *pixel_colors,            // [height, width, 3] - RGB color values per pixel
        float *gaussian_colors,               // [num_gaussians, 3] - output: accumulated colors per Gaussian
        int32_t *gaussian_hit_counts,         // [num_gaussians] - output: hit count per Gaussian
        int height,                           // Image height
        int width,                            // Image width
        int max_intersections,                // Maximum intersections per pixel
        int num_gaussians                     // Total number of Gaussians
    ) {
        // Calculate pixel coordinates
        int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
        int total_pixels = height * width;
        
        if (pixel_id >= total_pixels) return;
        
        // Get number of intersections for this pixel
        int num_hits = num_intersections[pixel_id];
        
        if (num_hits > 0) {
            // Get pixel color [R, G, B]
            const float *pixel_color = &pixel_colors[pixel_id * 3];
            
            // Process each intersection at this pixel
            for (int i = 0; i < num_hits && i < max_intersections; i++) {
                uint32_t gaussian_id = intersection_ids[pixel_id * max_intersections + i];
                
                if (gaussian_id < num_gaussians) {
                    // Use atomic operations to safely accumulate colors and counts
                    // Red channel
                    atomicAdd(&gaussian_colors[gaussian_id * 3 + 0], pixel_color[0]);
                    // Green channel  
                    atomicAdd(&gaussian_colors[gaussian_id * 3 + 1], pixel_color[1]);
                    // Blue channel
                    atomicAdd(&gaussian_colors[gaussian_id * 3 + 2], pixel_color[2]);
                    // Hit count
                    atomicAdd(&gaussian_hit_counts[gaussian_id], 1);
                }
            }
        }
    }

    __global__ void averageGaussianColors(
        float *gaussian_colors,               // [num_gaussians, 3] - accumulated colors (will be averaged in-place)
        const int32_t *gaussian_hit_counts,   // [num_gaussians] - hit counts
        int num_gaussians                     // Total number of Gaussians
    ) {
        int gaussian_id = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (gaussian_id >= num_gaussians) return;
        
        int32_t hit_count = gaussian_hit_counts[gaussian_id];
        
        if (hit_count > 0) {
            // Average the accumulated colors
            float inv_count = 1.0f / static_cast<float>(hit_count);
            gaussian_colors[gaussian_id * 3 + 0] *= inv_count;  // Red
            gaussian_colors[gaussian_id * 3 + 1] *= inv_count;  // Green
            gaussian_colors[gaussian_id * 3 + 2] *= inv_count;  // Blue
        }
    }


    extern "C" void convertQuaternionToRotationMatrix(float *quaternionsDevice, float *rotationMatricesDevice,
                                                      int numEllipsoids) {
        dim3 threadsPerBlock(128);
        const unsigned int numBlocks = (numEllipsoids + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 blocks(numBlocks);

        calculateRotationMatrices<<<blocks, threadsPerBlock>>>(
            quaternionsDevice, rotationMatricesDevice, numEllipsoids
        );
    }

    extern "C" void calc_rot_gradients(float *quaternions, float *normalized_quaternions, float *grad_rotations, float *grad_rotations_mat, int numEllipsoids){
        dim3 threadsPerBlock(128);
        const unsigned int numBlocks = (numEllipsoids + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 blocks(numBlocks);

        calc_rot_gradients_helper<<<blocks, threadsPerBlock>>>(
            quaternions, normalized_quaternions, grad_rotations, grad_rotations_mat, numEllipsoids
        );
    }

    extern "C" void calculateBoundingBoxes(float *centers, float *scales, float *rotations, float *boundingBoxes,
                                           int numEllipsoids) {
        dim3 threadsPerBlock(128);
        const unsigned int numBlocks = (numEllipsoids + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 blocks(numBlocks);

        calculateAABBs<<<blocks, threadsPerBlock>>>(
            centers, scales, rotations, boundingBoxes, numEllipsoids
        );
    };

    extern "C" void processIntersectionMedianColors(uint32_t *intersection_ids, uint32_t *num_intersections,
                                                   float *pixel_colors, float *gaussian_color_values,
                                                   float *gaussian_median_colors, int32_t *gaussian_hit_counts,
                                                   int height, int width, int max_intersections, 
                                                   int num_gaussians, int max_hits_per_gaussian) {
        // Launch kernel for processing intersections and storing all color values
        dim3 threadsPerBlock(128);
        const unsigned int numPixels = height * width;
        const unsigned int numBlocks = (numPixels + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 blocks(numBlocks);

        processIntersectionColorsForMedianKernel<<<blocks, threadsPerBlock>>>(
            intersection_ids, num_intersections, pixel_colors, gaussian_color_values, 
            gaussian_hit_counts, height, width, max_intersections, num_gaussians, max_hits_per_gaussian
        );
        
        // Launch kernel for calculating median colors
        const unsigned int numGaussianBlocks = (num_gaussians + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 gaussianBlocks(numGaussianBlocks);
        
        calculateMedianGaussianColors<<<gaussianBlocks, threadsPerBlock>>>(
            gaussian_color_values, gaussian_hit_counts, gaussian_median_colors, num_gaussians, max_hits_per_gaussian
        );
    }

    extern "C" void processIntersectionColors(uint32_t *intersection_ids, uint32_t *num_intersections,
                                             float *pixel_colors, float *gaussian_colors, 
                                             int32_t *gaussian_hit_counts,
                                             int height, int width, int max_intersections, int num_gaussians) {
        // Launch kernel for processing intersections
        dim3 threadsPerBlock(128);
        const unsigned int numPixels = height * width;
        const unsigned int numBlocks = (numPixels + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 blocks(numBlocks);

        processIntersectionColorsKernel<<<blocks, threadsPerBlock>>>(
            intersection_ids, num_intersections, pixel_colors, gaussian_colors, 
            gaussian_hit_counts, height, width, max_intersections, num_gaussians
        );
        
        // Launch kernel for averaging colors
        const unsigned int numGaussianBlocks = (num_gaussians + threadsPerBlock.x - 1) / threadsPerBlock.x;
        dim3 gaussianBlocks(numGaussianBlocks);
        
        averageGaussianColors<<<gaussianBlocks, threadsPerBlock>>>(
            gaussian_colors, gaussian_hit_counts, num_gaussians
        );
    }
}
