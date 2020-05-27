#pragma once

#include "typedefs.h"
#include "cuda_math.h"

struct Path
{
public:
	uint3 path;
	
	HOST_DEVICE Path(uint3 path)
		: path(path)
	{
	}
	HOST_DEVICE Path(uint32 x, uint32 y, uint32 z)
		: path(make_uint3(x, y, z))
	{
	}

	HOST_DEVICE void ascend(uint32 levels)
	{
		path.x >>= levels;
		path.y >>= levels;
		path.z >>= levels;
	}
	HOST_DEVICE void descend(uint8 child)
	{
		path.x <<= 1;
		path.y <<= 1;
		path.z <<= 1;
		path.x |= (child & 0x4u) >> 2;
		path.y |= (child & 0x2u) >> 1;
		path.z |= (child & 0x1u) >> 0;
	}

	HOST_DEVICE float3 as_position(uint32 extraShift = 0) const
	{
		return make_float3(
			float(path.x << extraShift),
			float(path.y << extraShift),
			float(path.z << extraShift)
		);
	}

	// level: level of the child!
	HOST_DEVICE uint8 child_index(uint32 level, uint32 totalLevels) const
	{
		check(level <= totalLevels);
		return uint8(
			(((path.x >> (totalLevels - level) & 0x1) == 0) ? 0 : 4) |
			(((path.y >> (totalLevels - level) & 0x1) == 0) ? 0 : 2) |
			(((path.z >> (totalLevels - level) & 0x1) == 0) ? 0 : 1));
	}

	HOST_DEVICE bool is_null() const
	{
		return path.x == 0 && path.y == 0 && path.z == 0;
	}

public:
	DEVICE static Path load(int32 x, int32 y, cudaSurfaceObject_t surface)
	{
#ifdef __CUDA_ARCH__
		Path path;
		path.path = make_uint3(surf2Dread<uint4>(surface, x * sizeof(uint4), y));
		return path;
#else
		check(false);
		return {};
#endif
	}
	DEVICE void store(int32 x, int32 y, cudaSurfaceObject_t surface)
	{
#ifdef __CUDA_ARCH__
		surf2Dwrite(make_uint4(path.x, path.y, path.z, 0), surface, x * sizeof(uint4), y);
#endif
	}

private:
	HOST_DEVICE Path() {}
};