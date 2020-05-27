#include "dag_tracer.h"
#include "cuda_error_check.h"
#include "memory.h"
#include "tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"

DAGTracer::DAGTracer(bool headLess)
	: headLess(headLess)
{
    if (headLess)
    {
        const auto setupArray = [](auto& array, auto& buffer, auto x, auto y, auto z, auto w)
        {
            cudaChannelFormatDesc desc = cudaCreateChannelDesc(x, y, z, w, cudaChannelFormatKindUnsigned);
            CUDA_CHECKED_CALL cudaMallocArray(&array, &desc, imageWidth, imageHeight, cudaArraySurfaceLoadStore);
            buffer.create_surface(array);
        };

        setupArray(pathArray, pathsBuffer, 32, 32, 32, 32);
        setupArray(colorsArray, colorsBuffer, 8, 8, 8, 8);
    }
    else
    {
        const auto setupImage = [](auto& buffer, auto& image, GLint formatA, GLenum formatB, GLenum formatC)
        {
            glGenTextures(1, &image);
            glBindTexture(GL_TEXTURE_2D, image);
            glTexImage2D(GL_TEXTURE_2D, 0, formatA, (int32)imageWidth, (int32)imageHeight, 0, formatB, formatC, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);
            buffer.register_resource(image);
        };

        setupImage(pathsBuffer, pathsImage, GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT);
        setupImage(colorsBuffer, colorsImage, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);

        pathCache = Memory::malloc<uint3>("path cache", sizeof(uint3), EMemoryType::GPU_Managed);
    }

	cudaEventCreate(&eventBeg);
	cudaEventCreate(&eventEnd);
}


DAGTracer::~DAGTracer()
{
    if (headLess)
    {
		pathsBuffer.destroy_surface();
		colorsBuffer.destroy_surface();
		cudaFreeArray(pathArray);
		cudaFreeArray(colorsArray);
    }
    else
    {
        pathsBuffer.unregister_resource();
        colorsBuffer.unregister_resource();
        glDeleteTextures(1, &pathsImage);
        glDeleteTextures(1, &colorsImage);

        Memory::free(pathCache);
    }

	cudaEventDestroy(eventBeg);
	cudaEventDestroy(eventEnd);
}

inline Tracer::TracePathsParams get_trace_params(
	const CameraView& camera, 
	uint32 levels,
	const DAGInfo& dagInfo)
{
	const double3 position  = make_double3(camera.position);
	const double3 direction = make_double3(camera.forward());
	const double3 up        = make_double3(camera.up());
	const double3 right     = make_double3(camera.right());

	const double3 boundsMin = make_double3(dagInfo.boundsAABBMin);
	const double3 boundsMax = make_double3(dagInfo.boundsAABBMax);

	const double fov = camera.fov / 2.0 * (double(M_PI) / 180.);
	const double aspect_ratio = double(imageWidth) / double(imageHeight);
	
	const double3 X = right     * sin(fov) * aspect_ratio;
	const double3 Y = up        * sin(fov);
	const double3 Z = direction * cos(fov);

	const double3 bottomLeft  = position + Z - Y - X;
	const double3 bottomRight = position + Z - Y + X;
	const double3 topLeft     = position + Z + Y - X;

	const double3 translation = -boundsMin;
	const double3 scale = make_double3(double(1 << levels)) / (boundsMax - boundsMin);

	const double3 finalPosition    = (position    + translation) * scale;
	const double3 finalBottomLeft  = (bottomLeft  + translation) * scale;
	const double3 finalTopLeft     = (topLeft     + translation) * scale;
	const double3 finalBottomRight = (bottomRight + translation) * scale;
	const double3 dx = (finalBottomRight - finalBottomLeft) * (1.0 / imageWidth);
	const double3 dy = (finalTopLeft     - finalBottomLeft) * (1.0 / imageHeight);

	Tracer::TracePathsParams params;

	params.cameraPosition = finalPosition;
	params.rayMin = finalBottomLeft;
	params.rayDDx = dx;
	params.rayDDy = dy;

	return params;
}


template<typename TDAG>
float DAGTracer::resolve_paths(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag)
{
	PROFILE_FUNCTION();
	
	const dim3 block_dim = dim3(4, 64);
	const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    if (!headLess) pathsBuffer.map_surface();
	auto traceParams = get_trace_params(camera, dag.levels, dagInfo);
	traceParams.pathsSurface = pathsBuffer.cudaSurface;

    CUDA_CHECK_ERROR();

	cudaEventRecord(eventBeg);
	Tracer::trace_paths <<<grid_dim, block_dim>>> (traceParams, dag);
	cudaEventRecord(eventEnd);
	cudaEventSynchronize(eventEnd);

	CUDA_CHECK_ERROR();

	float elapsed;
	cudaEventElapsedTime(&elapsed, eventBeg, eventEnd);
	CUDA_CHECK_ERROR();
	if (!headLess) pathsBuffer.unmap_surface();

	return elapsed;
}

template<typename TDAG, typename TDAGColors>
float DAGTracer::resolve_colors(const TDAG& dag, const TDAGColors& colors, EDebugColors debugColors, uint32 debugColorsIndexLevel, ToolInfo toolInfo)
{
	PROFILE_FUNCTION();
	
    colors.check_ready_for_rt();

	const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

	if (!headLess) pathsBuffer.map_surface();
	if (!headLess) colorsBuffer.map_surface();
	Tracer::TraceColorsParams traceParams;
	traceParams.debugColors = debugColors;
	traceParams.debugColorsIndexLevel = debugColorsIndexLevel;
	traceParams.toolInfo = toolInfo;
	traceParams.pathsSurface = pathsBuffer.cudaSurface;
	traceParams.colorsSurface = colorsBuffer.cudaSurface;

    CUDA_CHECK_ERROR();

	cudaEventRecord(eventBeg);
	Tracer::trace_colors<<<grid_dim, block_dim>>>(traceParams, dag, colors);
	cudaEventRecord(eventEnd);
	cudaEventSynchronize(eventEnd);

	float elapsed;
	cudaEventElapsedTime(&elapsed, eventBeg, eventEnd);
	CUDA_CHECK_ERROR();

	if (!headLess) pathsBuffer.unmap_surface();
	if (!headLess) colorsBuffer.unmap_surface();

	return elapsed;
}

template<typename TDAG>
float DAGTracer::resolve_shadows(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, float shadowBias, float fogDensity)
{
	PROFILE_FUNCTION();
	
    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    if (!headLess) pathsBuffer.map_surface();
	if (!headLess) colorsBuffer.map_surface();

	const auto pathParams = get_trace_params(camera, dag.levels, dagInfo);
    Tracer::TraceShadowsParams traceParams{
            pathParams.cameraPosition,
            pathParams.rayMin,
            pathParams.rayDDx,
            pathParams.rayDDy,
            shadowBias,
            fogDensity,
            pathsBuffer.cudaSurface,
            colorsBuffer.cudaSurface
    };

    CUDA_CHECK_ERROR();

	cudaEventRecord(eventBeg);
	Tracer::trace_shadows <<<grid_dim, block_dim>>> (traceParams, dag);
	cudaEventRecord(eventEnd);
	cudaEventSynchronize(eventEnd);

	float elapsed;
	cudaEventElapsedTime(&elapsed, eventBeg, eventEnd);
	CUDA_CHECK_ERROR();

	if (!headLess) pathsBuffer.unmap_surface();
	if (!headLess) colorsBuffer.unmap_surface();

	return elapsed;
}

template float DAGTracer::resolve_paths<BasicDAG>(const CameraView&, const DAGInfo&, const BasicDAG&);
template float DAGTracer::resolve_paths<HashDAG >(const CameraView&, const DAGInfo&, const HashDAG &);

template float DAGTracer::resolve_shadows<BasicDAG>(const CameraView&, const DAGInfo&, const BasicDAG&, float, float);
template float DAGTracer::resolve_shadows<HashDAG >(const CameraView&, const DAGInfo&, const HashDAG &, float, float);

#define COLORS_IMPL(Dag, Colors)\
template float DAGTracer::resolve_colors<Dag, Colors>(const Dag&, const Colors&, EDebugColors, uint32, ToolInfo);

COLORS_IMPL(BasicDAG, BasicDAGUncompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGCompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGColorErrors)
COLORS_IMPL(HashDAG, HashDAGColors)

__global__ void read_path(uint32 x, uint32 y, cudaSurfaceObject_t surface, uint3* output)
{
	*output = make_uint3(surf2Dread<uint4>(surface, x * sizeof(uint4), y));
}

uint3 DAGTracer::get_path(uint32 posX, uint32 posY)
{
	PROFILE_FUNCTION();
	
    if (headLess) return {};

	check(posX < imageWidth);
    check(posY < imageHeight);

    pathsBuffer.map_surface();
    CUDA_CHECK_ERROR();
	read_path<<<1,1>>>(posX, posY, pathsBuffer.cudaSurface, pathCache);
	CUDA_CHECK_ERROR();
    if (!headLess) pathsBuffer.unmap_surface();

    return *pathCache;
}
