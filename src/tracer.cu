#include "tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
	for (uint8 child = 0; child < 8; ++child)
	{
		uint8 childInOrder = child ^ order;
		if (mask & (1u << childInOrder))
			return childInOrder;
	}
	check(false);
	return 0;
}

template<bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(
	uint32 level,
	const Path& path,
	const TDAG& dag,
	const float3& rayOrigin,
	const float3& rayDirection,
	const float3& rayDirectionInverted)
{
	// Find node center = .5 * (boundsMin + boundsMax) + .5f
	const uint32 shift = dag.levels - level;

	const float radius = float(1u << (shift - 1));
	const float3 center = make_float3(radius) + path.as_position(shift);

	const float3 centerRelativeToRay = center - rayOrigin;

	// Ray intersection with axis-aligned planes centered on the node
	// => rayOrg + tmid * rayDir = center
	const float3 tmid = centerRelativeToRay * rayDirectionInverted;

	// t-values for where the ray intersects the slabs centered on the node
	// and extending to the side of the node
	float tmin, tmax;
	{
		const float3 slabRadius = radius * abs(rayDirectionInverted);
		const float3 pmin = tmid - slabRadius;
		tmin = max(max(pmin), .0f);

		const float3 pmax = tmid + slabRadius;
		tmax = min(pmax);
	}

	// Check if we actually hit the root node
	// This test may not be entirely safe due to float precision issues.
	// especially on lower levels. For the root node this seems OK, though.
	if (isRoot && (tmin >= tmax))
	{
		return 0;
	}

	// Identify first child that is intersected
	// NOTE: We assume that we WILL hit one child, since we assume that the
	//       parents bounding box is hit.
	// NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
	//       intersection point, since this point might lie too close to an
	//       axis plane. Instead, we use the midpoint between max and min which
	//       will lie in the correct node IF the ray only intersects one node.
	//       Otherwise, it will still lie in an intersected node, so there are
	//       no false positives from this.
	uint8 intersectionMask = 0;
	{
		const float3 pointOnRay = (0.5f * (tmin + tmax)) * rayDirection;

		uint8 const firstChild =
			((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) +
			((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) +
			((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

		intersectionMask |= (1u << firstChild);
	}

	// We now check the points where the ray intersects the X, Y and Z plane.
	// If the intersection is within (ray_tmin, ray_tmax) then the intersection
	// point implies that two voxels will be touched by the ray. We find out
	// which voxels to mask for an intersection point at +X, +Y by setting
	// ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
	//
	// NOTE: When the intersection point is close enough to another axis plane,
	//       we must check both sides or we will get robustness issues.
	const float epsilon = 1e-4f;

	if (tmin <= tmid.x && tmid.x <= tmax)
	{
		const float3 pointOnRay = tmid.x * rayDirection;

		uint8 A = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) A |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) A |= 0x33;

		uint8 B = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) B |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) B |= 0x55;

		intersectionMask |= A & B;
	}
	if (tmin <= tmid.y && tmid.y <= tmax)
	{
		const float3 pointOnRay = tmid.y * rayDirection;

		uint8 C = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) C |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) C |= 0x0F;

		uint8 D = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) D |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) D |= 0x55;

		intersectionMask |= C & D;
	}
	if (tmin <= tmid.z && tmid.z <= tmax)
	{
		const float3 pointOnRay = tmid.z * rayDirection;

		uint8 E = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) E |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) E |= 0x0F;


		uint8 F = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) F |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) F |= 0x33;

		intersectionMask |= E & F;
	}

	return intersectionMask;
}

struct StackEntry
{
	uint32 index;
	uint8 childMask;
	uint8 visitMask;
};

template<typename TDAG>
__global__ void Tracer::trace_paths(const TracePathsParams traceParams, const TDAG dag)
{
	// Target pixel coordinate
	const uint2 pixel = make_uint2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (pixel.x >= imageWidth || pixel.y >= imageHeight)
		return; // outside.

	// Pre-calculate per-pixel data
	const float3 rayOrigin = make_float3(traceParams.cameraPosition);
	const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));

	const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));
	const uint8 rayChildOrder =
		(rayDirection.x < 0.f ? 4 : 0) +
		(rayDirection.y < 0.f ? 2 : 0) +
		(rayDirection.z < 0.f ? 1 : 0);

	// State
	uint32 level = 0;
	Path path(0, 0, 0);

	StackEntry stack[MAX_LEVELS];
	StackEntry cache;
	Leaf cachedLeaf; // needed to iterate on the last few levels

	cache.index = dag.get_first_node_index();
	cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
	cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

	// Traverse DAG
	for (;;)
	{
		// Ascend if there are no children left.
		{
			uint32 newLevel = level;
			while (newLevel > 0 && !cache.visitMask)
			{
				newLevel--;
				cache = stack[newLevel];
			}

			if (newLevel == 0 && !cache.visitMask)
			{
				path = Path(0, 0, 0);
				break;
			}

			path.ascend(level - newLevel);
			level = newLevel;
		}

		// Find next child in order by the current ray's direction
		const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);

		// Mark it as handled
		cache.visitMask &= ~(1u << nextChild);

		// Intersect that child with the ray
		{
			path.descend(nextChild);
			stack[level] = cache;
			level++;

			// If we're at the final level, we have intersected a single voxel.
			if (level == dag.levels)
			{
				break;
			}

			// Are we in an internal node?
			if (level < dag.leaf_level())
			{
				cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
				cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
				cache.visitMask = cache.childMask &	compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
			else
			{
				/* The second-to-last and last levels are different: the data
				 * of these two levels (2^3 voxels) are packed densely into a
				 * single 64-bit word.
				 */
				uint8 childMask;

				if (level == dag.leaf_level())
				{
					const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
					cachedLeaf = dag.get_leaf(addr);
					childMask = cachedLeaf.get_first_child_mask();
				}
				else
				{
					childMask = cachedLeaf.get_second_child_mask(nextChild);
				}

				// No need to set the index for bottom nodes
				cache.childMask = childMask;
				cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
		}
	}

	path.store(pixel.x, imageHeight - 1 - pixel.y, traceParams.pathsSurface);
}

template<typename TDAG, typename TDAGColors>
__global__ void Tracer::trace_colors(const TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors)
{
	const uint2 pixel = make_uint2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (pixel.x >= imageWidth || pixel.y >= imageHeight)
		return; // outside

	const auto setColorImpl = [&](uint32 color)
	{
		surf2Dwrite(color, traceParams.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
	};

	const Path path = Path::load(pixel.x, pixel.y, traceParams.pathsSurface);
	if (path.is_null())
    {
        setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f));
        return;
	}

	const float toolStrength = traceParams.toolInfo.strength(path);
	const auto setColor = [&](uint32 color)
	{
#if TOOL_OVERLAY
		if (toolStrength > 0)
		{
			color = ColorUtils::float3_to_rgb888(lerp(ColorUtils::rgb888_to_float3(color), make_float3(1, 0, 0), clamp(100 * toolStrength, 0.f, .5f)));
		}
#endif
        setColorImpl(color);
    };

    const auto invalidColor = [&]()
    {
        uint32 b = (path.path.x ^ path.path.y ^ path.path.z) & 0x1;
        setColor(ColorUtils::float3_to_rgb888(make_float3(1, b, 1.f - b)));
    };

    uint64 nof_leaves = 0;
	uint32 debugColorsIndex = 0;

	uint32 colorNodeIndex = 0;
	typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();

	uint32 level = 0;
	uint32 nodeIndex = dag.get_first_node_index();
	while (level < dag.leaf_level())
	{
		level++;

		// Find the current childmask and which subnode we are in
		const uint32 node = dag.get_node(level - 1, nodeIndex);
		const uint8 childMask = Utils::child_mask(node);
		const uint8 child = path.child_index(level, dag.levels);

		// Make sure the node actually exists
		if (!(childMask & (1 << child)))
		{
			setColor(0xFF00FF);
			return;
		}

		ASSUME(level > 0);
		if (level - 1 < colors.get_color_tree_levels())
		{
			colorNodeIndex = colors.get_child_index(level - 1, colorNodeIndex, child);
			if (level == colors.get_color_tree_levels())
			{
				check(nof_leaves == 0);
				colorLeaf = colors.get_leaf(colorNodeIndex);
			}
			else
			{
				// TODO nicer interface
				if (!colorNodeIndex)
				{
					invalidColor();
					return;
				}
			}
		}

		// Debug
		if (traceParams.debugColors == EDebugColors::Index ||
			traceParams.debugColors == EDebugColors::Position ||
			traceParams.debugColors == EDebugColors::ColorTree)
		{
			if (traceParams.debugColors == EDebugColors::Index &&
				traceParams.debugColorsIndexLevel == level - 1)
			{
				debugColorsIndex = nodeIndex;
			}
			if (level == dag.leaf_level())
			{
				if (traceParams.debugColorsIndexLevel == dag.leaf_level())
				{
					check(debugColorsIndex == 0);
					const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
					debugColorsIndex = childIndex;
				}

				if (traceParams.debugColors == EDebugColors::Index)
				{
					setColor(Utils::murmurhash32(debugColorsIndex));
				}
				else if (traceParams.debugColors == EDebugColors::Position)
				{
					constexpr uint32 checkerSize = 0x7FF;
					float color = ((path.path.x ^ path.path.y ^ path.path.z) & checkerSize) / float(checkerSize);
					color = (color + 0.5) / 2;
					setColor(ColorUtils::float3_to_rgb888(Utils::has_flag(nodeIndex) ? make_float3(color, 0, 0) : make_float3(color)));
				}
				else
				{
					check(traceParams.debugColors == EDebugColors::ColorTree);
					const uint32 offset = dag.levels - colors.get_color_tree_levels();
					const float color = ((path.path.x >> offset) ^ (path.path.y >> offset) ^ (path.path.z >> offset)) & 0x1;
					setColor(ColorUtils::float3_to_rgb888(make_float3(color)));
				}
				return;
			}
			else
			{
				nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
				continue;
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// Find out how many leafs are in the children preceding this
		//////////////////////////////////////////////////////////////////////////
		// If at final level, just count nof children preceding and exit
		if (level == dag.leaf_level())
		{
			for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
			{
				if (childMask & (1u << childBeforeChild))
				{
					const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
					const Leaf leaf = dag.get_leaf(childIndex);
					nof_leaves += Utils::popcll(leaf.to_64());
				}
			}
			const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
			const Leaf leaf = dag.get_leaf(childIndex);
			const uint8 leafBitIndex =
				(((path.path.x & 0x1) == 0) ? 0 : 4) |
				(((path.path.y & 0x1) == 0) ? 0 : 2) |
				(((path.path.z & 0x1) == 0) ? 0 : 1) |
				(((path.path.x & 0x2) == 0) ? 0 : 32) |
				(((path.path.y & 0x2) == 0) ? 0 : 16) |
				(((path.path.z & 0x2) == 0) ? 0 : 8);
			nof_leaves += Utils::popcll(leaf.to_64() & ((uint64(1) << leafBitIndex) - 1));

			break;
		}
		else
		{
			ASSUME(level > 0);
			if (level > colors.get_color_tree_levels())
			{
				// Otherwise, fetch the next node (and accumulate leaves we pass by)
				for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
				{
					if (childMask & (1u << childBeforeChild))
					{
						const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
						const uint32 childNode = dag.get_node(level, childIndex);
						nof_leaves += colors.get_leaves_count(level, childNode);
					}
				}
			}
			nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
		}
	}

	if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves))
	{
	    invalidColor();
		return;
	}

	auto compressedColor = colorLeaf.get_color(nof_leaves);
	uint32 color =
		traceParams.debugColors == EDebugColors::ColorBits
		? compressedColor.get_debug_hash()
		: ColorUtils::float3_to_rgb888(
			traceParams.debugColors == EDebugColors::MinColor
			? compressedColor.get_min_color()
			: traceParams.debugColors == EDebugColors::MaxColor
			? compressedColor.get_max_color()
			: traceParams.debugColors == EDebugColors::Weight
			? make_float3(compressedColor.get_weight())
			: compressedColor.get_color());
	setColor(color);
}

template<typename TDAG>
inline __device__ bool intersect_ray_node_out_of_order(const TDAG& dag, const float3 rayOrigin, const float3 rayDirection)
{
    const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));

	// State
	uint32 level = 0;
	Path path(0, 0, 0);

	StackEntry stack[MAX_LEVELS];
	StackEntry cache;
	Leaf cachedLeaf; // needed to iterate on the last few levels

	cache.index = dag.get_first_node_index();
	cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
	cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

	// Traverse DAG
	for (;;)
	{
		// Ascend if there are no children left.
		{
			uint32 newLevel = level;
			while (newLevel > 0 && !cache.visitMask)
			{
				newLevel--;
				cache = stack[newLevel];
			}

			if (newLevel == 0 && !cache.visitMask)
			{
				path = Path(0, 0, 0);
				break;
			}

			path.ascend(level - newLevel);
			level = newLevel;
		}

		// Find next child in order by the current ray's direction
		const uint8 nextChild = 31 - __clz(cache.visitMask);

		// Mark it as handled
		cache.visitMask &= ~(1u << nextChild);

		// Intersect that child with the ray
		{
			path.descend(nextChild);
			stack[level] = cache;
			level++;

			// If we're at the final level, we have intersected a single voxel.
			if (level == dag.levels)
			{
			    return true;
			}

			// Are we in an internal node?
			if (level < dag.leaf_level())
			{
				cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
				cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
				cache.visitMask = cache.childMask &	compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
			else
			{
				/* The second-to-last and last levels are different: the data
				 * of these two levels (2^3 voxels) are packed densely into a
				 * single 64-bit word.
				 */
				uint8 childMask;

				if (level == dag.leaf_level())
				{
					const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
					cachedLeaf = dag.get_leaf(addr);
					childMask = cachedLeaf.get_first_child_mask();
				}
				else
				{
					childMask = cachedLeaf.get_second_child_mask(nextChild);
				}

				// No need to set the index for bottom nodes
				cache.childMask = childMask;
				cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
		}
	}
	return false;
}

// Directed towards the sun
HOST_DEVICE float3 sun_direction()
{
    return normalize(make_float3(0.3f, 1.f, 0.5f));
}

HOST_DEVICE float3 applyFog(float3 rgb,      // original color of the pixel
                            double distance, // camera to point distance
                            double3 rayDir,   // camera to point vector
                            double3 rayOri,
                            float fogDensity)  // camera position
{
#if 0
    constexpr float fogDensity = 0.0001f;
    constexpr float c = 1.f;
    constexpr float heightOffset = 20000.f;
    constexpr float heightScale = 1.f;
    double fogAmount = c * exp((heightOffset - rayOri.y * heightScale) * fogDensity) * (1.0 - exp(-distance * rayDir.y * fogDensity)) / rayDir.y;
#else
    fogDensity *= 0.00001f;
    double fogAmount = 1.0 - exp(-distance * fogDensity);
#endif
    double sunAmount = 1.01f * max(dot(rayDir, make_double3(sun_direction())), 0.0);
    float3 fogColor = lerp(make_float3(187, 242, 250) / 255.f, // blue
                              make_float3(1.0f), // white
                              float(pow(sunAmount, 30.0)));
    return lerp(rgb, fogColor, clamp(float(fogAmount), 0.f, 1.f));
}

HOST_DEVICE double3 ray_box_intersection(double3 orig, double3 dir, double3 box_min, double3 box_max)
{
    double3 tmin = (box_min - orig) / dir;
    double3 tmax = (box_max - orig) / dir;

    double3 real_min = min(tmin, tmax);
    double3 real_max = max(tmin, tmax);

    // double minmax = min(min(real_max.x, real_max.y), real_max.z);
    double maxmin = max(max(real_min.x, real_min.y), real_min.z);

    // checkf(minmax >= maxmin, "%f > %f", minmax, maxmin);
    return orig + dir * maxmin;
}

template<typename TDAG>
__global__ void Tracer::trace_shadows(const TraceShadowsParams params, const TDAG dag)
{
    const uint2 pixel = make_uint2(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const auto setColorImpl = [&](float3 color)
    {
        const uint32 finalColor = ColorUtils::float3_to_rgb888(color);
        surf2Dwrite(finalColor, params.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
    };
    const auto setColor = [&](float light, double distance, double3 direction)
    {
        const uint32 colorInt = surf2Dread<uint32>(params.colorsSurface, pixel.x * sizeof(uint32), pixel.y);
        float3 color = ColorUtils::rgb888_to_float3(colorInt);

        color = color * clamp(0.5f + light, 0.f, 1.f);

        color = applyFog(
                color,
                distance,
                direction,
                params.cameraPosition,
                params.fogDensity);

        setColorImpl(color);
    };

    const float3 rayOrigin = make_float3(Path::load(pixel.x, pixel.y, params.pathsSurface).path);
    const double3 cameraRayDirection = normalize(params.rayMin + pixel.x * params.rayDDx + (imageHeight - 1 - pixel.y) * params.rayDDy - params.cameraPosition);

#if EXACT_SHADOWS || PER_VOXEL_FACE_SHADING
    const double3 rayOriginDouble = make_double3(rayOrigin);
    const double3 hitPosition = ray_box_intersection(
            params.cameraPosition,
            cameraRayDirection,
            rayOriginDouble,
            rayOriginDouble + 1);
#endif

#if EXACT_SHADOWS
    const float3 shadowStart = make_float3(hitPosition);
#else
    const float3 shadowStart = rayOrigin;
#endif

#if 0
    setColorImpl(make_float3(clamp_vector(normal, 0, 1)));
    return;
#endif

    if (length(rayOrigin) == 0.0f)
    {
        setColor(1, 1e9, cameraRayDirection);
        return; // Discard cleared or light-backfacing fragments
    }

    const float3 direction = sun_direction();
    const bool isShadowed = intersect_ray_node_out_of_order(dag, shadowStart + params.shadowBias * direction, direction);

    const double3 v = make_double3(rayOrigin) - params.cameraPosition;
    const double distance = length(v);
    const double3 nv = v / distance;

    if (isShadowed)
    {
        setColor(0, distance, nv);
    }
    else
    {
#if PER_VOXEL_FACE_SHADING
        const double3 voxelOriginToHitPosition = normalize(hitPosition - (rayOriginDouble + 0.5));
        const auto truncate_signed = [](double3 d) { return make_double3(int32(d.x), int32(d.y), int32(d.z)); };
        const double3 normal = truncate_signed(voxelOriginToHitPosition / max(abs(voxelOriginToHitPosition)));
        setColor(max(0.f, dot(make_float3(normal), sun_direction())), distance, nv);
#else
        setColor(1, distance, nv);
#endif
    }

#if 0 // AO code copy-pasted from Erik's impl, doesn't compile at all
    constexpr int sqrtNofSamples = 8;

    float avgSum = 0;
    for (int y = 0; y < sqrtNofSamples; y++)
    {
        for (int x = 0; x < sqrtNofSamples; x++)
        {
            int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
            float3 normal = make_float3(tex2D(normalTexture, float(coord.x), float(coord.y)));
            float3 tangent = normalize3(perp3(normal));
            float3 bitangent = cross(normal, tangent);
            //int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x)%RAND_SIZE, (coord.y * sqrtNofSamples + y + randomSeed.y)%RAND_SIZE);
            int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x) & RAND_BITMASK, (coord.y * sqrtNofSamples + y + randomSeed.y) & RAND_BITMASK);
            float2 randomSample = tex2D(randomTexture, randomCoord.x, randomCoord.y);
            float randomLength = tex2D(randomTexture, randomCoord.y, randomCoord.x).x;
            float2 dxdy = make_float2(1.0f / float(sqrtNofSamples), 1.0f / float(sqrtNofSamples));
            float3 sample = cosineSampleHemisphere(make_float2(x * dxdy.x, y * dxdy.y) + (1.0 / float(sqrtNofSamples)) * randomSample);
            float3 ray_d = normalize3(sample.x * tangent + sample.y * bitangent + sample.z * normal);
            avgSum += intersectRayNode_outOfOrder<maxLevels>(ray_o, ray_d, ray_tmax * randomLength, rootCenter, rootRadius, coneOpening) ? 0.0f : 1.0f;
        }
    }
    avgSum /= float(sqrtNofSamples * sqrtNofSamples);
#endif
}

template __global__ void Tracer::trace_paths<BasicDAG>(TracePathsParams, BasicDAG);
template __global__ void Tracer::trace_paths<HashDAG >(TracePathsParams, HashDAG);

template __global__ void Tracer::trace_shadows<BasicDAG>(TraceShadowsParams, BasicDAG);
template __global__ void Tracer::trace_shadows<HashDAG >(TraceShadowsParams, HashDAG);

#define COLORS_IMPL(Dag, Colors)\
template __global__ void Tracer::trace_colors<Dag, Colors>(TraceColorsParams, Dag, Colors);

COLORS_IMPL(BasicDAG, BasicDAGUncompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGCompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGColorErrors)
COLORS_IMPL(HashDAG, HashDAGColors)