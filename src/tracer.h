#pragma once

#include "cuda_math.h"
#include "typedefs.h"
#include "path.h"

enum class EDebugColors
{
	None,
	Index,
	Position,
	ColorTree,
	ColorBits,
	MinColor,
	MaxColor,
	Weight
};

constexpr uint32 CNumDebugColors = 8;

enum class ETool
{
	Sphere,
	SpherePaint,
	SphereNoise,
	Cube,
	CubeCopy,
	CubeFill,
};

constexpr uint32 CNumTools = 6;

struct ToolInfo
{
	ETool tool;
	Path position = Path(0, 0, 0);
	float radius;
	Path copySource = Path(0, 0, 0);
	Path copyDest = Path(0, 0, 0);

	ToolInfo() = default;
	ToolInfo(ETool tool, uint3 position, float radius, uint3 copySource, uint3 copyDest)
		: tool(tool)
		, position(position)
		, radius(radius)
		, copySource(copySource)
		, copyDest(copyDest)
	{
	}

	HOST_DEVICE float strength(const Path path) const
	{
		switch (tool)
		{
		case ETool::Sphere:
		case ETool::SpherePaint:
		case ETool::SphereNoise:
			return sphere_strength(position, path, radius);
		case ETool::Cube:
			return cube_strength(position, path, radius);
		case ETool::CubeCopy:
		case ETool::CubeFill:
		default:
			return max(max(sphere_strength(copySource, path, 3), sphere_strength(copyDest, path, 3)), cube_strength(position, path, radius));
		}
	}

private:
	HOST_DEVICE static float cube_strength(const Path pos, const Path path, float radius)
	{
		return 1 - max(abs(pos.as_position() - path.as_position())) / radius;
	}
	HOST_DEVICE static  float sphere_strength(const Path pos, const Path path, float radius)
	{
		return 1 - length(pos.as_position() - path.as_position()) / radius;
	}
};

namespace Tracer
{
	struct TracePathsParams
	{
		// In
		double3 cameraPosition;
		double3 rayMin;
		double3 rayDDx;
		double3 rayDDy;

		// Out
		cudaSurfaceObject_t pathsSurface;
	};

	template<typename TDAG>
	__global__ void trace_paths(const TracePathsParams traceParams, const TDAG dag);

	struct TraceColorsParams
	{
		TraceColorsParams() = default;

		// In
		EDebugColors debugColors;
		uint32 debugColorsIndexLevel;

		ToolInfo toolInfo;

		cudaSurfaceObject_t pathsSurface;

		// Out
		cudaSurfaceObject_t colorsSurface;
	};

	template<typename TDAG, typename TDAGColors>
    __global__ void trace_colors(const TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors);

    struct TraceShadowsParams
    {
        TraceShadowsParams() = default;

        // In
		double3 cameraPosition;
		double3 rayMin;
		double3 rayDDx;
		double3 rayDDy;
        float shadowBias;
        float fogDensity;
        cudaSurfaceObject_t pathsSurface;

        // In/Out
        cudaSurfaceObject_t colorsSurface;
    };

    template<typename TDAG>
    __global__ void trace_shadows(const TraceShadowsParams params, const TDAG dag);
}