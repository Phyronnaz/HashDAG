#pragma once

#include "typedefs.h"
#include "tracer.h"
#include "camera_view.h"
#include "cuda_gl_buffer.h"
#include "dag_info.h"

class DAGTracer
{
public:
	const bool headLess;

	DAGTracer(bool headLess);
	~DAGTracer();

	inline GLuint get_colors_image() const
	{
		return colorsImage;
	}

	template<typename TDAG>
	float resolve_paths(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag);
	template<typename TDAG, typename TDAGColors>
	float resolve_colors(const TDAG& dag, const TDAGColors& colors, EDebugColors debugColors, uint32 debugColorsIndexLevel, ToolInfo toolInfo);
	template<typename TDAG>
	float resolve_shadows(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, float shadowBias, float fogDensity);

	uint3 get_path(uint32 posX, uint32 posY);

private:
	GLuint pathsImage = 0;
	GLuint colorsImage = 0;

	CudaGLBuffer pathsBuffer;
	CudaGLBuffer colorsBuffer;

	cudaArray* pathArray = nullptr;
	cudaArray* colorsArray = nullptr;

	uint3* pathCache = nullptr;
	cudaEvent_t eventBeg, eventEnd;
};