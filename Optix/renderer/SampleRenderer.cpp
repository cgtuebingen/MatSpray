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
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

namespace osc
{
    extern "C" char embedded_ptx_code[];
    extern "C" char embedded_ptx_code2[];

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        GaussianSBTData data;
    };

    SampleRenderer::SampleRenderer(Model *model)
        : model(model), usingIntrinsics(false), lastFx(0.0f), lastFy(0.0f), lastCx(0.0f), lastCy(0.0f)
    {
        initOptix();

        std::cout << "#osc: creating optix context ..." << std::endl;
        createContext();

        std::cout << "#osc: setting up module ..." << std::endl;
        createModule();

        std::cout << "#osc: creating raygen programs ..." << std::endl;
        createRaygenPrograms();
        std::cout << "#osc: creating miss programs ..." << std::endl;
        createMissPrograms();
        std::cout << "#osc: creating hitgroup programs ..." << std::endl;
        createHitgroupPrograms();

        launchParams.traversable = buildAccel();

        std::cout << "#osc: setting up optix pipeline ..." << std::endl;
        createPipeline();

        std::cout << "#osc: building SBT ..." << std::endl;
        buildSBT();

        launchParamsBuffer.alloc(sizeof(launchParams));
        std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

        std::cout << GDT_TERMINAL_GREEN;
        std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
        std::cout << GDT_TERMINAL_DEFAULT;
    }

    OptixTraversableHandle SampleRenderer::buildAccel()
    {
        CUDA_SYNC_CHECK();

        OptixTraversableHandle asHandle{0};

        numEllipsoidsBuffer.free();
        numEllipsoidsBuffer.alloc_and_upload(model->numEllipsoids);

        CUDA_SYNC_CHECK();

        // AABB inputs
        OptixBuildInput aabbInput;
        CUdeviceptr d_aabb;
        uint32_t aabbInputFlags;

        aabbBufferSingle.d_ptr = model->aabbs;
        d_aabb = aabbBufferSingle.d_pointer();

        aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        aabbInput.customPrimitiveArray.aabbBuffers = &d_aabb;
        aabbInput.customPrimitiveArray.numPrimitives = model->numEllipsoids;
        aabbInputFlags = 0;
        aabbInput.customPrimitiveArray.flags = &aabbInputFlags;
        aabbInput.customPrimitiveArray.numSbtRecords = 1;
        aabbInput.customPrimitiveArray.strideInBytes = OPTIX_AABB_BUFFER_BYTE_ALIGNMENT *
                                                        ((sizeof(OptixAabb) + OPTIX_AABB_BUFFER_BYTE_ALIGNMENT - 1) / OPTIX_AABB_BUFFER_BYTE_ALIGNMENT);
        aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        aabbInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        aabbInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

        // BLAS setup
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                                 &accelOptions,
                                                 &aabbInput,
                                                 1,
                                                 &blasBufferSizes));

        // Prepare compaction
        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        // Build
        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(optixContext,
                                    /* stream */ 0,
                                    &accelOptions,
                                    &aabbInput,
                                    1,
                                    tempBuffer.d_pointer(),
                                    tempBuffer.sizeInBytes,
                                    outputBuffer.d_pointer(),
                                    outputBuffer.sizeInBytes,
                                    &asHandle,
                                    &emitDesc, 1));
        CUDA_SYNC_CHECK();

        // Compact
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);

        cudaFree(asBuffer.d_ptr);
        asBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(optixContext,
                                      /*stream:*/ 0,
                                      asHandle,
                                      asBuffer.d_pointer(),
                                      asBuffer.sizeInBytes,
                                      &asHandle));
        CUDA_SYNC_CHECK();

        outputBuffer.free();
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
    }

    void SampleRenderer::initOptix()
    {
        std::cout << "#osc: initializing optix..." << std::endl;

        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("#osc: no CUDA capable devices found!");
        std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

        OPTIX_CHECK(optixInit());
        std::cout << GDT_TERMINAL_GREEN
                  << "#osc: successfully initialized optix... yay!"
                  << GDT_TERMINAL_DEFAULT << std::endl;
    }

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *)
    {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
    }

    void SampleRenderer::createContext()
    {
        const int deviceID = 0;
        CUDA_CHECK(SetDevice(deviceID));
        CUDA_CHECK(StreamCreate(&stream));

        cudaGetDeviceProperties(&deviceProps, deviceID);
        std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

        CUresult cuRes = cuCtxGetCurrent(&cudaContext);
        if (cuRes != CUDA_SUCCESS)
            fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
    }

    void SampleRenderer::createModule()
    {
        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "SLANG_globalParams";

        pipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode_slang = embedded_ptx_code2;

        char log[2048];
        size_t sizeof_log = sizeof(log);
#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(optixModuleCreate(optixContext,
                                      &moduleCompileOptions,
                                      &pipelineCompileOptions,
                                      ptxCode.c_str(),
                                      ptxCode.size(),
                                      log, &sizeof_log,
                                      &module));
#else
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                             &moduleCompileOptions,
                                             &pipelineCompileOptions,
                                             ptxCode_slang.c_str(),
                                             ptxCode_slang.size(),
                                             log,
                                             &sizeof_log,
                                             &module));
#endif
        if (sizeof_log > 1)
            PRINT(log);
    }

    void SampleRenderer::createRaygenPrograms()
    {
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &raygenPGs[0]));
        if (sizeof_log > 1)
            PRINT(log);
    }

    void SampleRenderer::createMissPrograms()
    {
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = module;
        pgDesc.miss.entryFunctionName = "__miss__miss_radiance";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &missPGs[0]));
        if (sizeof_log > 1)
            PRINT(log);
    }

    void SampleRenderer::createHitgroupPrograms()
    {
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = module;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__closesthit_radiance";
        pgDesc.hitgroup.moduleAH = module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__anyhit_radiance";
        pgDesc.hitgroup.moduleIS = module;
        pgDesc.hitgroup.entryFunctionNameIS = "__intersection__ellipsoid_intersection";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &hitgroupPGs[0]));
        if (sizeof_log > 1)
            PRINT(log);
    }

    void SampleRenderer::createPipeline()
    {
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs) programGroups.push_back(pg);
        for (auto pg : missPGs) programGroups.push_back(pg);
        for (auto pg : hitgroupPGs) programGroups.push_back(pg);

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pipelineCompileOptions,
                                        &pipelineLinkOptions,
                                        programGroups.data(),
                                        (int)programGroups.size(),
                                        log, &sizeof_log,
                                        &pipeline));
        if (sizeof_log > 1)
            PRINT(log);

        pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        OPTIX_CHECK(optixPipelineSetStackSize(
                                              pipeline,
                                              2 * 1024,
                                              2 * 1024,
                                              2 * 1024,
                                              1));
        if (sizeof_log > 1)
            PRINT(log);
    }

    void SampleRenderer::buildSBT()
    {
        // Raygen record
        RaygenRecord raygenRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[0], &raygenRecord));
        raygenRecord.data = nullptr;

        raygenRecordsBuffer.alloc_and_upload(raygenRecord);
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

        // Miss records
        std::vector<MissRecord> missRecords;
        for (int i = 0; i < missPGs.size(); i++)
        {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr;
            missRecords.push_back(rec);
        }
        missRecordsBuffer.alloc_and_upload(missRecords);
        sbt.missRecordBase = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = (int)missRecords.size();

        // Hitgroup records
        HitgroupRecord hitgroupRecords{};
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &hitgroupRecords));

        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = 1;
    }

    void SampleRenderer::render()
    {
        if (launchParams.frame.size.x == 0)
            return;

        launchParamsBuffer.upload(&launchParams, 1);

        OPTIX_CHECK(optixLaunch(pipeline, stream,
                                launchParamsBuffer.d_pointer(),
                                launchParamsBuffer.sizeInBytes,
                                &sbt,
                                launchParams.frame.size.x,
                                launchParams.frame.size.y,
                                1));
        CUDA_SYNC_CHECK();
    }

    void SampleRenderer::setCamera(const Camera &camera)
    {
        lastSetCamera = camera;
        launchParams.camera.position = camera.from;
        launchParams.camera.direction = normalize(camera.at);
        
        usingIntrinsics = false;
        
        const float camera_angle_x = 0.6911112070083618f;
        const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
        
        const float tanHalfFovX = tan(camera_angle_x * 0.5f);
        const float tanHalfFovY = tanHalfFovX / aspect;
        
        launchParams.camera.horizontal = 2.0f * tanHalfFovX * normalize(cross(camera.up, launchParams.camera.direction));
        launchParams.camera.vertical = 2.0f * tanHalfFovY * normalize(camera.up);
    }

    void SampleRenderer::setCameraFromIntrinsics(const Camera &camera,
                                                 float focalLengthX, float focalLengthY,
                                                 float principalPointX, float principalPointY)
    {
        lastSetCamera = camera;
        launchParams.camera.position = camera.from;
        launchParams.camera.direction = normalize(camera.at);

        usingIntrinsics = true;
        lastFx = focalLengthX;
        lastFy = focalLengthY;
        lastCx = principalPointX;
        lastCy = principalPointY;

        const float width  = float(launchParams.frame.size.x);
        const float height = float(launchParams.frame.size.y);
        const float fx = fmaxf(focalLengthX, 1e-6f);
        const float fy = fmaxf(focalLengthY, 1e-6f);
        const float tanHalfFovX = 0.5f * width  / fx;
        const float tanHalfFovY = 0.5f * height / fy;

        launchParams.camera.horizontal = 2.0f * tanHalfFovX * normalize(cross(camera.up, launchParams.camera.direction));
        launchParams.camera.vertical   = 2.0f * tanHalfFovY * normalize(camera.up);
    }

    void SampleRenderer::resize(const vec2i &newSize)
    {
        if (newSize.x == 0 | newSize.y == 0)
            return;

        const unsigned int max_num_intersections = 150;

        numIntersectionsBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
        intGaussianIdsBuffer.resize(newSize.x * newSize.y * max_num_intersections * sizeof(uint32_t));
        prev_indexBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
        depthBuffer.resize(newSize.x * newSize.y * sizeof(float));
        
        cudaMemset(numIntersectionsBuffer.d_ptr, 0, newSize.x * newSize.y * sizeof(uint32_t));
        cudaMemset(intGaussianIdsBuffer.d_ptr, 0, newSize.x * newSize.y * max_num_intersections * sizeof(uint32_t));
        cudaMemset(prev_indexBuffer.d_ptr, 0, newSize.x * newSize.y * sizeof(uint32_t));
        cudaMemset(depthBuffer.d_ptr, 0, newSize.x * newSize.y * sizeof(float));
        
        launchParams.frame.size = newSize;
        launchParams.centers.data = (float *)meansBuffer.d_ptr;
        launchParams.centers.size = model->numEllipsoids * 3;
        launchParams.scales.data = (float *)scalesBuffer.d_ptr;
        launchParams.scales.size = model->numEllipsoids * 3;
        launchParams.rotations.data = (float *)rotationsBuffer.d_ptr;
        launchParams.rotations.size = model->numEllipsoids * 9;
        launchParams.densities.data = (float *)densitiesBuffer.d_ptr;
        launchParams.densities.size = model->numEllipsoids;
        launchParams.intGaussianIdsBuffer.data = (uint32_t *)intGaussianIdsBuffer.d_pointer();
        launchParams.intGaussianIdsBuffer.size = newSize.x * newSize.y * max_num_intersections;
        launchParams.numIntersectionsBuffer.data = (uint32_t *)numIntersectionsBuffer.d_pointer();
        launchParams.numIntersectionsBuffer.size = newSize.x * newSize.y;
        launchParams.prev_indexBuffer.data = (uint32_t *)prev_indexBuffer.d_pointer();
        launchParams.prev_indexBuffer.size = newSize.x * newSize.y;
        launchParams.depthBuffer.data = (float *)depthBuffer.d_pointer();
        launchParams.depthBuffer.size = newSize.x * newSize.y;
        launchParams.epsilon = 1e-6f;
        launchParams.max_num_intersections = max_num_intersections;
        launchParams.numEllipsoids = model->numEllipsoids;
        launchParams.depth_max_tolerance = 0.01f;

        if (usingIntrinsics) {
            setCameraFromIntrinsics(lastSetCamera, lastFx, lastFy, lastCx, lastCy);
        } else {
            setCamera(lastSetCamera);
        }
    }

    void SampleRenderer::downloadPixels(float *cpu_buffer, size_t buffer_size)
    {
        if (colorBuffer.d_ptr == nullptr) {
            throw std::runtime_error("Color buffer is not allocated");
        }
        const size_t expected = size_t(launchParams.frame.size.x) * size_t(launchParams.frame.size.y) * 3;
        if (buffer_size > expected) {
            throw std::runtime_error("Buffer size too large for available color data");
        }
        cudaMemcpy(cpu_buffer, colorBuffer.d_ptr,
                   buffer_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void SampleRenderer::updateParameters(float *centers,
                                        float *scales,
                                        float *rotationMats,
                                        float *densities,
                                        const int num_ellipsoids,
                                        const bool allocate_new)
    {
        if (allocate_new){
            cudaFree(meansBuffer.d_ptr);
            cudaFree(scalesBuffer.d_ptr);
            cudaFree(rotationsBuffer.d_ptr);
            cudaFree(densitiesBuffer.d_ptr);

            cudaMalloc(&meansBuffer.d_ptr, num_ellipsoids * 3 * sizeof(float));
            cudaMalloc(&scalesBuffer.d_ptr, num_ellipsoids * 3 * sizeof(float));
            cudaMalloc(&rotationsBuffer.d_ptr, num_ellipsoids * 9 * sizeof(float));
            cudaMalloc(&densitiesBuffer.d_ptr, num_ellipsoids * sizeof(float));
        }

        cudaMemcpy(meansBuffer.d_ptr, centers, num_ellipsoids * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(scalesBuffer.d_ptr, scales, num_ellipsoids * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(rotationsBuffer.d_ptr, rotationMats, num_ellipsoids * 9 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(densitiesBuffer.d_ptr, densities, num_ellipsoids * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    void SampleRenderer::downloadIntersectionData(uint32_t* cpu_buffer, size_t buffer_size) const {
        if (intGaussianIdsBuffer.d_ptr == nullptr) {
            throw std::runtime_error("Intersection buffer is not allocated");
        }
        if (buffer_size > intGaussianIdsBuffer.sizeInBytes / sizeof(uint32_t)) {
            throw std::runtime_error("Buffer size too large for available data");
        }
        cudaMemcpy(cpu_buffer, intGaussianIdsBuffer.d_ptr, 
                   buffer_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    void SampleRenderer::downloadNumIntersectionsData(uint32_t* cpu_buffer, size_t buffer_size) const {
        if (numIntersectionsBuffer.d_ptr == nullptr) {
            throw std::runtime_error("Num intersections buffer is not allocated");
        }
        if (buffer_size > numIntersectionsBuffer.sizeInBytes / sizeof(uint32_t)) {
            throw std::runtime_error("Buffer size too large for available data");
        }
        cudaMemcpy(cpu_buffer, numIntersectionsBuffer.d_ptr, 
                   buffer_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    void SampleRenderer::setDepthBufferDevice(float *d_depth, size_t num_values)
    {
        const size_t expected = size_t(launchParams.frame.size.x) * size_t(launchParams.frame.size.y);
        if (expected == 0) return;
        if (num_values != expected) {
            throw std::runtime_error("Depth buffer size mismatch in setDepthBufferDevice");
        }
        cudaMemcpy(depthBuffer.d_ptr, d_depth, expected * sizeof(float), cudaMemcpyDeviceToDevice);
    }
} // ::osc
