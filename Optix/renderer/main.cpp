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

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <random>
#include "helperFunctionsDevice.h"
#include <string>
#include <sstream>
#include <array>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    // struct SampleWindow : public GLFCameraWindow {
    //     SampleWindow(const std::string &title,
    //                  const Model *model,
    //                  const Camera &camera,
    //                  const float worldScale)
    //         : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
    //           sample(model) {
    //         sample.setCamera(camera);
    //     }

    //     virtual void render() override {
    //         if (cameraFrame.modified) {
    //             sample.setCamera(Camera{
    //                 cameraFrame.get_from(),
    //                 cameraFrame.get_at(),
    //                 cameraFrame.get_up()
    //             });
    //             cameraFrame.modified = false;
    //         }
    //         sample.render();
    //     }

    //     virtual void draw() override {
    //         glEnable(GL_BLEND);
    //         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //         sample.downloadPixels(pixels.data());
    //         if (fbTexture == 0)
    //             glGenTextures(1, &fbTexture);

    //         glBindTexture(GL_TEXTURE_2D, fbTexture);
    //         GLenum texFormat = GL_RGBA;
    //         GLenum texelType = GL_UNSIGNED_BYTE;
    //         glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
    //                      texelType, pixels.data());

    //         glDisable(GL_LIGHTING);
    //         glColor3f(1, 1, 1);

    //         glMatrixMode(GL_MODELVIEW);
    //         glLoadIdentity();

    //         glEnable(GL_TEXTURE_2D);
    //         glBindTexture(GL_TEXTURE_2D, fbTexture);
    //         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //         glDisable(GL_DEPTH_TEST);

    //         glViewport(0, 0, fbSize.x, fbSize.y);

    //         glMatrixMode(GL_PROJECTION);
    //         glLoadIdentity();
    //         glOrtho(0.f, (float) fbSize.x, 0.f, (float) fbSize.y, -1.f, 1.f);

    //         glBegin(GL_QUADS); {
    //             glTexCoord2f(0.f, 0.f);
    //             glVertex3f(0.f, 0.f, 0.f);

    //             glTexCoord2f(0.f, 1.f);
    //             glVertex3f(0.f, (float) fbSize.y, 0.f);

    //             glTexCoord2f(1.f, 1.f);
    //             glVertex3f((float) fbSize.x, (float) fbSize.y, 0.f);

    //             glTexCoord2f(1.f, 0.f);
    //             glVertex3f((float) fbSize.x, 0.f, 0.f);
    //         }
    //         glEnd();
    //     }

    //     virtual void resize(const vec2i &newSize) {
    //         fbSize = newSize;
    //         sample.resize(newSize);
    //         pixels.resize(newSize.x * newSize.y);
    //     }

    //     vec2i fbSize;
    //     GLuint fbTexture{0};
    //     SampleRenderer sample;
    //     std::vector<uint32_t> pixels;
    // };

    template <typename T>
    std::vector<T> read_matrix(std::ifstream &file, int &rows, int &cols)
    {
        file.read(reinterpret_cast<char *>(&rows), sizeof(rows)); // Read the number of rows
        file.read(reinterpret_cast<char *>(&cols), sizeof(cols)); // Read the number of columns

        std::cout << "Matrix size: " << rows << "x" << cols << std::endl; // Debug print

        if (rows <= 0 || cols <= 0)
        {
            // Add sanity checks
            std::cerr << "Invalid size: " << rows << "x" << cols << std::endl;
            return std::vector<T>(); // Return empty vector if size is invalid
        }

        std::vector<T> matrix(rows * cols);                                          // Allocate memory for the matrix
        file.read(reinterpret_cast<char *>(matrix.data()), rows * cols * sizeof(T)); // Read the data
        return matrix;
    }

    class Gaussians
    {
    public:
        int active_sh_degree;
        float spatial_lr_scale;
        std::vector<float> _xyz;
        std::vector<float> _normal;
        std::vector<float> _shs_dc;
        std::vector<float> _shs_rest;
        std::vector<float> _scaling;
        std::vector<float> _rotation;
        std::vector<float> _opacity;

        // Rows and columns for each matrix
        int rows_xyz, cols_xyz;
        int rows_normal, cols_normal;
        int rows_shs_dc, cols_shs_dc;
        int rows_shs_rest, cols_shs_rest;
        int rows_scaling, cols_scaling;
        int rows_rotation, cols_rotation;
        int rows_opacity, cols_opacity;

        void read_from_binary(const std::string &file_path)
        {
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Could not open the file!" << std::endl;
                return;
            }

            // Read scalar values
            file.read(reinterpret_cast<char *>(&active_sh_degree), sizeof(active_sh_degree));
            file.read(reinterpret_cast<char *>(&spatial_lr_scale), sizeof(spatial_lr_scale));

            // Read matrices (rows, columns first, then data)
            _xyz = read_matrix<float>(file, rows_xyz, cols_xyz);
            _normal = read_matrix<float>(file, rows_normal, cols_normal);
            _shs_dc = read_matrix<float>(file, rows_shs_dc, cols_shs_dc); // Assuming same size
            _shs_rest = read_matrix<float>(file, rows_shs_rest, cols_shs_rest);
            _scaling = read_matrix<float>(file, rows_scaling, cols_scaling);
            _rotation = read_matrix<float>(file, rows_rotation, cols_rotation);
            // For rotation, the second dimension is likely different
            _opacity = read_matrix<float>(file, rows_opacity, cols_opacity);

            file.close();
        }

        void print_sizes() const
        {
            std::cout << "Size of _xyz: " << _xyz.size() << std::endl;
            std::cout << "Size of _normal: " << rows_normal << "x" << cols_normal << std::endl;
            std::cout << "Size of _rotation: " << rows_rotation << "x" << cols_rotation << std::endl;
            std::cout << "Size of _shs_dc: " << rows_shs_dc << "x" << cols_shs_dc << std::endl;
            std::cout << "Size of _shs_rest: " << rows_shs_rest << "x" << cols_shs_rest << std::endl;
        }
    };

    void quaternion_to_rotation_matrix(const std::array<float, 4> &q, float rotation_matrix[3][3])
    {
        float w = q[0];
        float x = q[1];
        float y = q[2];
        float z = q[3];

        rotation_matrix[0][0] = 1 - 2 * (y * y + z * z);
        rotation_matrix[0][1] = 2 * (x * y - w * z);
        rotation_matrix[0][2] = 2 * (x * z + w * y);

        rotation_matrix[1][0] = 2 * (x * y + w * z);
        rotation_matrix[1][1] = 1 - 2 * (x * x + z * z);
        rotation_matrix[1][2] = 2 * (y * z - w * x);

        rotation_matrix[2][0] = 2 * (x * z - w * y);
        rotation_matrix[2][1] = 2 * (y * z + w * x);
        rotation_matrix[2][2] = 1 - 2 * (x * x + y * y);
    }

    std::vector<Ellipsoid> convert_gaussians_to_ellipsoids(const std::vector<std::array<float, 3>> &means,
                                                           const std::vector<std::array<float, 3>> &scales,
                                                           const std::vector<std::array<std::array<float, 3>, 3>> &
                                                               rotations,
                                                           const std::vector<std::array<std::array<float, 3>, 16>> &
                                                               shs,
                                                           const std::vector<float> &opacities,
                                                           const std::vector<aabb> &boundingBoxes)
    {
        std::vector<Ellipsoid> ellipsoids;
        ellipsoids.reserve(means.size());

        for (size_t i = 0; i < means.size(); ++i)
        {
            Ellipsoid _ellipsoid;
            _ellipsoid.center = {means[i][0], means[i][1], means[i][2]};
            _ellipsoid.scale = {scales[i][0], scales[i][1], scales[i][2]}; // Adjust based on your scaling tensor
            //_ellipsoid.scale = {0.015f, 0.015f, 0.015f};
            std::memcpy(_ellipsoid.rotation, rotations[i].data(), sizeof(rotations[i]));
            //std::memcpy(_ellipsoid.sh, shs[i].data(), sizeof(shs[i]));
            _ellipsoid.opacity = opacities[i];
            _ellipsoid.boundingBox = boundingBoxes[i];

            // std::cout << "bounding box: " << _ellipsoid.boundingBox.lower << " " << _ellipsoid.boundingBox.upper << std::endl;

            ellipsoids.emplace_back(_ellipsoid);
        }

        return ellipsoids;
    }

    void setIdentityMatrix(float matrix[3][3])
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i][j] = (i == j) ? 1.0f : 0.0f; // Identity matrix
            }
        }
    }

    void calcRotsAndBounds(std::vector<float> &centers,
                           std::vector<float> &scaling,
                           std::vector<float> &quaternions,
                           std::vector<float> &rotationMatrices,
                           std::vector<float> &boundingBoxes,
                           int numEllipsoids)
    {
        float *centersDevice;
        float *scalingDevice;
        float *quaternionsDevice;
        float *rotationMatricesDevice;
        float *boundingBoxesDevice;

        cudaMalloc(&centersDevice,
                   numEllipsoids * 3 * sizeof(float));
        cudaMalloc(&scalingDevice,
                   numEllipsoids * 3 * sizeof(float));
        cudaMalloc(&quaternionsDevice,
                   numEllipsoids * 4 * sizeof(float));
        cudaMalloc(&rotationMatricesDevice,
                   numEllipsoids * 3 * 3 * sizeof(float));
        cudaMalloc(&boundingBoxesDevice,
                   numEllipsoids * 3 * 2 * sizeof(float));

        cudaMemcpy(centersDevice, centers.data(),
                   numEllipsoids * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(scalingDevice, scaling.data(),
                   numEllipsoids * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(quaternionsDevice, quaternions.data(),
                   numEllipsoids * 4 * sizeof(float),
                   cudaMemcpyHostToDevice);

        convertQuaternionToRotationMatrix(quaternionsDevice, rotationMatricesDevice, numEllipsoids);

        // Check for any CUDA errors
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        }

        calculateBoundingBoxes(centersDevice, scalingDevice, rotationMatricesDevice, boundingBoxesDevice,
                               numEllipsoids);

        // Check for any CUDA errors
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(rotationMatrices.data(), rotationMatricesDevice,
                   numEllipsoids * 3 * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(boundingBoxes.data(), boundingBoxesDevice,
                   numEllipsoids * 3 * 2 * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(centersDevice);
        cudaFree(scalingDevice);
        cudaFree(quaternionsDevice);
        cudaFree(rotationMatricesDevice);
        cudaFree(boundingBoxesDevice);
    }

    extern "C" int main()
    {
        //         std::random_device rd; // Seed for the random number engine
        //         std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        //         std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        //         std::uniform_real_distribution<float> dis2(.0f, 1.0f);
        //
        //
        //         Gaussians gaussians;
        //
        //         // Path to the binary file
        //         std::string binary_file_path = "../models/lego_3dgs_30000.bin";
        //
        //         // // Load the binary file
        //         gaussians.read_from_binary(binary_file_path);
        //
        //         // Output some of the loaded data to verify
        //         gaussians.print_sizes();
        //
        //         std::vector<float> rotationsHost(9 * gaussians.rows_rotation);
        //         std::vector<float> hostBoundingBoxes(6 * gaussians.rows_xyz);
        //
        //         calcRotsAndBounds(gaussians._xyz, gaussians._scaling, gaussians._rotation,
        //                           rotationsHost, hostBoundingBoxes, gaussians.rows_rotation);
        //
        //         std::vector<std::array<float, 3> > means;
        //         std::vector<std::array<float, 3> > scales;
        //         std::vector<std::array<std::array<float, 3>, 3> > rotations;
        //         std::vector<std::array<std::array<float, 3>, 16> > shs;
        //         std::vector<float> opacities;
        //         std::vector<aabb> boundingBoxes;
        //
        //         shs.reserve(means.size());
        //         const int rest_sh_cols = gaussians.cols_shs_rest / 3;
        //
        //         for (int i = 0; i < gaussians.rows_xyz; ++i) {
        //             means.push_back({gaussians._xyz[i * 3], gaussians._xyz[i * 3 + 1], gaussians._xyz[i * 3 + 2]});
        //             scales.push_back({
        //                 sqrt(3.f) * gaussians._scaling[i * 3], sqrt(3.f) * gaussians._scaling[i * 3 + 1],
        //                 sqrt(3.f) * gaussians._scaling[i * 3 + 2]
        //             });
        //             std::array<std::array<float, 3>, 3> rotation = {
        //                 {
        //                     {rotationsHost[i * 9], rotationsHost[i * 9 + 1], rotationsHost[i * 9 + 2]},
        //                     {rotationsHost[i * 9 + 3], rotationsHost[i * 9 + 4], rotationsHost[i * 9 + 5]},
        //                     {rotationsHost[i * 9 + 6], rotationsHost[i * 9 + 7], rotationsHost[i * 9 + 8]}
        //                 }
        //             };
        //             rotations.push_back(rotation);
        //             std::array<std::array<float, 3>, 16> harmonic{};
        //
        //             std::array<float, 3> color_dc{};
        //             color_dc[0] = gaussians._shs_dc[i * 3];
        //             color_dc[1] = gaussians._shs_dc[i * 3 + 1];
        //             color_dc[2] = gaussians._shs_dc[i * 3 + 2];
        //             harmonic[0] = color_dc;
        //
        //             for (int j = 0; j < rest_sh_cols; ++j) {
        //                 std::array<float, 3> color;
        //                 color[0] = gaussians._shs_rest[i * gaussians.cols_shs_rest + j * 3];
        //                 color[1] = gaussians._shs_rest[i * gaussians.cols_shs_rest + j * 3 + 1];
        //                 color[2] = gaussians._shs_rest[i * gaussians.cols_shs_rest + j * 3 + 2];
        //                 harmonic[j + 1] = color;
        //             }
        //             shs.emplace_back(harmonic);
        //
        //             opacities.push_back(gaussians._opacity[i]);
        //
        //             aabb bounds;
        //             bounds.lower = {hostBoundingBoxes[i * 6], hostBoundingBoxes[i * 6 + 1], hostBoundingBoxes[i * 6 + 2]};
        //             bounds.upper = {
        //                 hostBoundingBoxes[i * 6 + 3], hostBoundingBoxes[i * 6 + 4], hostBoundingBoxes[i * 6 + 5]
        //             };
        //             boundingBoxes.push_back(bounds);
        //
        //         }
        //
        //         std::vector<Ellipsoid> ellipsoids = convert_gaussians_to_ellipsoids(
        //             means, scales, rotations, shs, opacities, boundingBoxes);
        //
        //         std::array<float, 3> mean = calculateMean(means);
        //
        //         // Print the mean
        //         std::cout << "Mean: (" << mean[0] << ", " << mean[1] << ", " << mean[2] << ")\n";
        //
        //         // std::vector<Ellipsoid> ellipsoids;
        //         //
        //         // for (int i = 0; i < 10000; i++) {
        //         //     Ellipsoid ellipsoid;
        //         //     ellipsoid.center = {dis(gen), dis(gen), dis(gen)};
        //         //     ellipsoid.scale = {100.f, 10.f, 10.f};
        //         //
        //         //     float u1 = dis2(gen);
        //         //     float u2 = dis2(gen);
        //         //     float u3 = dis2(gen);
        //         //
        //         //     // Step 3: Compute the quaternion values
        //         //     float w = std::sqrt(1.0f - u1) * std::sin(2.0f * M_PI * u2);
        //         //     float x = std::sqrt(1.0f - u1) * std::cos(2.0f * M_PI * u2);
        //         //     float y = std::sqrt(u1) * std::sin(2.0f * M_PI * u3);
        //         //     float z = std::sqrt(u1) * std::cos(2.0f * M_PI * u3);
        //         //
        //         //     std::array<float, 4> quaternion[4] = {w, x, y, z};
        //         //
        //         //     quaternion_to_rotation_matrix(quaternion[0], ellipsoid.rotation);
        //         //     ellipsoids.emplace_back(ellipsoid);
        //         // }
        //
        //         try {
        //             Model *model = addEllipsoids(ellipsoids);
        //
        //             model->shDegree = gaussians.active_sh_degree;
        //
        //             Camera camera = {
        //                 /*from*/vec3f(3.f, -2.5f, 2.f),
        //                 /* at */model->bounds.center(),
        //                 /* up */vec3f(0.f, 0.f, 1.f)
        //             };
        //             // something approximating the scale of the world, so the
        //             // camera knows how much to move for any given user interaction:
        //             const float worldScale = length(model->bounds.span());
        //
        //             SampleWindow *window = new SampleWindow("Optix 7 Course Example",
        //                                                     model, camera, worldScale);
        //             window->run();
        //         } catch (std::runtime_error &e) {
        //             std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
        //                     << GDT_TERMINAL_DEFAULT << std::endl;
        //             std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" <<
        //                     std::endl;
        //             exit(1);
        //         }
        //
        //         return 0;
    }
} // ::osc
