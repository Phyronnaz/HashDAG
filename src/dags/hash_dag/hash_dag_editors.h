#pragma once

#include "typedefs.h"
#include "color_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/dag_utils.h"
#include "FastNoise.h"

template<typename TChild>
struct Editor
{
    inline bool should_edit(const Path path, uint32 depth) const
    {
        const float3 start = path.as_position(depth);
        return self().should_edit_impl(start, start + make_float3(float(1u << depth)));
    }

    inline bool is_full(const Path path, uint32 depth) const
    {
        const float3 start = path.as_position(depth);
        return self().is_full_impl(start, start + make_float3(float(1u << depth)));
    }
    inline bool is_empty(const Path path, uint32 depth) const
    {
        const float3 start = path.as_position(depth);
        return self().is_empty_impl(start, start + make_float3(float(1u << depth)));
    }

	inline bool is_full_impl(float3 inMin, float3 inMax) const
	{
        return false;
	}
	inline bool is_empty_impl(float3 inMin, float3 inMax) const
	{
        return false;
	}

	// Used when is_full is true
	inline float3 get_single_color() const
    {
        check(false);
        return {};
    }

	inline uint32 edit_count() const
	{
		return ~uint32(0);
	}

private:
    inline const TChild& self() const
    {
        return static_cast<const TChild&>(*this);
    }
};

struct BoxEditorBase
{
	const float radius;
	const float3 min;
	const float3 max;

	// can be edited by children
	float3 boundsMin;
	float3 boundsMax;

	BoxEditorBase(float3 center, float radius)
		: radius(radius)
		, min(center - make_float3(radius))
		, max(center + make_float3(radius))
	{
		boundsMin = min;
		boundsMax = max;
	}

	inline bool should_edit_impl(float3 inMin, float3 inMax) const
	{
		return !(
			boundsMin.x >= inMax.x ||
			boundsMin.y >= inMax.y ||
			boundsMin.z >= inMax.z ||
			boundsMax.x <= inMin.x ||
			boundsMax.y <= inMin.y ||
			boundsMax.z <= inMin.z ||
			inMin.x >= boundsMax.x ||
			inMin.y >= boundsMax.y ||
			inMin.z >= boundsMax.z ||
			inMax.x <= boundsMin.x ||
			inMax.y <= boundsMin.y ||
			inMax.z <= boundsMin.z);
	}
	inline uint32 edit_count_impl() const
	{
		auto const ext = boundsMax-boundsMin;
		return uint32(ext.x)*uint32(ext.y)*uint32(ext.z);
	}
};

struct SphereEditorBase
{
	const float3 center;
	const float radius;
	const float radiusSquared;

	SphereEditorBase(float3 center, float radius)
		: center(center)
		, radius(radius)
		, radiusSquared(radius * radius)
	{
	}

	inline bool should_edit_impl(float3 min, float3 max) const
	{
		float dist = radiusSquared;

		if (center.x < min.x) dist -= squared(center.x - min.x);
		else if (center.x > max.x) dist -= squared(center.x - max.x);

		if (center.y < min.y) dist -= squared(center.y - min.y);
		else if (center.y > max.y) dist -= squared(center.y - max.y);

		if (center.z < min.z) dist -= squared(center.z - min.z);
		else if (center.z > max.z) dist -= squared(center.z - max.z);

		return dist > 0;
	}
};

struct BufferEditor : BoxEditorBase
{
	const HashDAG dag;
	const uint32 size;

	enum EValues : uint8
    {
	    Unknown = 0,
        LoadedTrue = 1,
        LoadedFalse = 2,
        SetTrue = 3,
        SetFalse = 4
    };
	EValues* __restrict__ const values;

	BufferEditor(const HashDAG& dag, float3 center, float radius)
		: BoxEditorBase(center, radius)
		, dag(dag)
		, size(2 * uint32(std::ceil(radius)))
		, values(Memory::malloc<EValues>("editor tool buffer", sizeof(EValues) * uint64(size) * uint64(size) * uint64(size), EMemoryType::CPU))
	{
        printf("Size: %u; Allocating %f MB\n", size, Utils::to_MB(uint64(size) * uint64(size) * uint64(size) * sizeof(uint8)));
        std::memset(values, EValues::Unknown, size * size * size);
	}
	~BufferEditor()
	{
		Memory::free(values);
	}
	BufferEditor(const BufferEditor&) = delete;

	inline Path get_world_path(uint32 x, uint32 y, uint32 z) const
    {
        return { uint32(min.x + float(x)),
                 uint32(min.y + float(y)),
                 uint32(min.z + float(z)) };
    }

	inline bool is_valid(uint32 x, uint32 y, uint32 z) const
	{
		return x < size && y < size && z < size;
	}
	inline uint32 is_valid(uint3 p) const
	{
		return is_valid(p.x, p.y, p.z);
	}

	inline uint64 get_index(uint32 x, uint32 y, uint32 z) const
	{
		check(is_valid(x, y, z));
		return uint64(x) + uint64(size) * uint64(y) + uint64(size) * uint64(size) * uint64(z);
	}
	inline uint64 get_index(uint3 p) const
	{
		return get_index(p.x, p.y, p.z);
	}

	inline bool get_value(uint32 x, uint32 y, uint32 z) const
	{
        const uint64 index = get_index(x, y, z);
        if (values[index] == EValues::Unknown)
        {
            values[index] = DAGUtils::get_value(dag, get_world_path(x, y, z))
                            ? EValues::LoadedTrue
                            : EValues::LoadedFalse;
        }
        return values[index] == EValues::LoadedTrue || values[index] == EValues::SetTrue;
	}

	inline bool get_value(uint3 p) const
	{
		return get_value(p.x, p.y, p.z);
	}

	inline void set_value(uint32 x, uint32 y, uint32 z, bool value) const
    {
        values[get_index(x, y, z)] = value ? EValues::SetTrue : EValues::SetFalse;
    }
	inline void set_value(uint3 p, bool value) const
	{
		return set_value(p.x, p.y, p.z, value);
	}

	inline bool get_new_value(float3 position, bool oldValue) const
	{
        const float3 fp = position - min;
        const uint3 p = truncate(fp);
        if (!is_valid(p) || (get_value(p) == EValues::Unknown))
        {
            return oldValue;
        }
        return get_value(p);
	}
};

template<bool isAdding>
struct BoxEditor final : BoxEditorBase, Editor<BoxEditor<isAdding>>
{
	BoxEditor(float3 center, float radius)
		: BoxEditorBase(center, radius)
	{
		boundsMin = min;
		boundsMax = max;
	}

    inline bool is_full_impl(float3 inMin, float3 inMax) const
    {
        return isAdding &&
               boundsMin.x <= inMin.x &&
               boundsMin.y <= inMin.y &&
               boundsMin.z <= inMin.z &&
               boundsMax.x >= inMax.x &&
               boundsMax.y >= inMax.y &&
               boundsMax.z >= inMax.z;
    }
    inline bool is_empty_impl(float3 inMin, float3 inMax) const
    {
        return !isAdding &&
               boundsMin.x <= inMin.x &&
               boundsMin.y <= inMin.y &&
               boundsMin.z <= inMin.z &&
               boundsMax.x >= inMax.x &&
               boundsMax.y >= inMax.y &&
               boundsMax.z >= inMax.z;
    }
	inline constexpr float3 get_single_color() const
    {
	    return { 1, 0, 1 };
    }

	inline bool get_new_value(float3 position, bool oldValue) const
	{
		check(should_edit_impl(position, position + make_float3(1)));
		return isAdding;
	}
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		CompressedColor color;
		color.set_single_color({ 1, 0, 1 });
		return color;
	}
};

template<bool isAdding>
struct SphereEditor final : SphereEditorBase, Editor<SphereEditor<isAdding>>
{
	SphereEditor(float3 center, float radius)
		: SphereEditorBase(center, radius)
	{
	}

    inline bool is_full_impl(float3 min, float3 max) const
    {
        return isAdding &&
               ::max(squared(min.x - center.x), squared(max.x - center.x)) +
               ::max(squared(min.y - center.y), squared(max.y - center.y)) +
               ::max(squared(min.z - center.z), squared(max.z - center.z))
               < radiusSquared;
    }
    inline bool is_empty_impl(float3 min, float3 max) const
    {
        return !isAdding &&
               ::max(squared(min.x - center.x), squared(max.x - center.x)) +
               ::max(squared(min.y - center.y), squared(max.y - center.y)) +
               ::max(squared(min.z - center.z), squared(max.z - center.z))
               < radiusSquared;
    }
	inline constexpr float3 get_single_color() const
    {
	    return { 1, 0, 0 };
    }

	inline bool get_new_value(float3 position, bool oldValue) const
	{
		check(should_edit_impl(position, position + make_float3(1)));
		return isAdding;
	}
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		CompressedColor color;
		color.set_single_color(get_single_color());
		return color;
	}
};

struct SpherePaintEditor final : SphereEditorBase, Editor<SpherePaintEditor>
{
	using SphereEditorBase::SphereEditorBase;

	inline bool get_new_value(float3 position, bool oldValue) const
	{
		return oldValue;
	}
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		CompressedColor color;
		color.set_single_color(lerp(oldColor().get_color(), make_float3(0, 0, 1), clamp(1 - length(position - center) / radius, 0.f, .1f)));
		return color;
	}
};

// Queries every voxel individually
template<bool withTransform>
struct CopyEditorWithoutDecompression final : BoxEditorBase, Editor<CopyEditorWithoutDecompression<withTransform>>
{
	// Copy of the dag as we want to access the old one
	const HashDAG dagCopy;
	// But ref of the colors in case the arrays are reallocated cuz too small
	const HashDAGColors& colors;
	const float3 source;
	const float3 dest;

	const Matrix3x3 transform;
	const Matrix3x3 inverseTransform;

	const bool withSwirl;
	const float swirlPeriod;

	CopyEditorWithoutDecompression(HashDAG dagCopy, const HashDAGColors& colors, float3 source, float3 dest, float3 center, float radius, const Matrix3x3& transform, StatsRecorder& statsRecorder, bool withSwirl, float swirlPeriod)

		: BoxEditorBase(dest + (withTransform ? make_float3(transform * Vector3(center - source)) : center - source), withTransform ? radius * (transform * Vector3(1)).abs().largest() : radius)
		, dagCopy(dagCopy)
		, colors(colors)
		, source(source)
		, dest(dest)
		, transform(transform)
		, inverseTransform(Matrix3x3::Inverse(transform))
		, withSwirl(withSwirl)
		, swirlPeriod(swirlPeriod)
	{
	}

	inline float3 destination_to_source(float3 position) const
	{
		float3 relativePosition = position - dest;
		if (withSwirl && COPY_CAN_APPLY_SWIRL)
		{
			const float angle = relativePosition.y / swirlPeriod;
			const float x = relativePosition.x;
			const float z = relativePosition.z;
			relativePosition.x = x * cos(angle) - z * sin(angle);
			relativePosition.z = x * sin(angle) + z * cos(angle);
		}
		if (withTransform)
		{
			relativePosition = make_float3(inverseTransform * Vector3(relativePosition));
		}
		return source + relativePosition;
	}

#if COPY_EMPTY_CHECKS
    inline bool should_edit(const Path path, uint32 depth) const
    {
        if (!Editor<CopyEditorWithoutDecompression<withTransform>>::should_edit(path, depth))
        {
            return false;
        }
		if (withTransform) // Can't do a clean check with it
		{
			return true;
		}
		
        const uint32 level = C_maxNumberOfLevels - depth;
        if (level < (C_maxNumberOfLevels-2))
        {
            const float3 start = path.as_position(depth);
            const float3 offset = start - dest + source;
            if (offset.x >= 0 && offset.y >= 0 && offset.z >= 0)
            {
                const uint3 position = truncate(offset);
                if (DAGUtils::is_empty(dagCopy, level, position, make_uint3(1u << depth)))
                {
                    return false;
                }
            }
        }
        return true;
    }
	inline uint32 edit_count() const
	{
		return BoxEditorBase::edit_count_impl();
	}
#endif

	inline bool get_new_value(float3 position, bool oldValue) const
	{
		if (oldValue)
		{
			return oldValue;
		}
		const float3 p = destination_to_source(position);
		return DAGUtils::get_value(dagCopy, Path(uint32(p.x), uint32(p.y), uint32(p.z)));
	}
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		if (oldValue == newValue || !newValue)
		{
			return oldColor();
		}

		const float3 p = destination_to_source(position);
		return DAGUtils::get_color(dagCopy, colors, Path(uint32(p.x), uint32(p.y), uint32(p.z)));
	}
};

// Use a bunch copy
struct CopyEditorWithDecompression final : BoxEditorBase, Editor<CopyEditorWithDecompression>
{
	// Copy of the dag as we want to access the old one
	//const HashDAG dagCopy;
	// For WithDecompression it's *probably* safe to skip the copy.
	const HashDAG& dagCopy;
	// But ref of the colors in case the arrays are reallocated cuz too small
	const HashDAGColors& dagColors;
	const float3 source;
	const float3 dest;
	const uint32 size;
	const float3 copyMin;
    const uint3 intCopyMin;

	bool* __restrict__ const values;
#if BENCHMARK
    CompressedColor* __restrict__ const colors;
#endif

	CopyEditorWithDecompression(const HashDAG& dag, const HashDAGColors& dagColors, float3 source, float3 dest, float3 center, float radius, const Matrix3x3& transform, StatsRecorder& statsRecorder, bool withSwirl, float swirlPeriod)
		: BoxEditorBase(center - source + dest, radius)
		, dagCopy(dag)
		, dagColors(dagColors)
		, source(source)
		, dest(dest)
		, size(2 * uint32(std::ceil(radius)))
		, copyMin(center - make_float3(radius))
		, intCopyMin(truncate(copyMin))
		, values(Memory::malloc<bool>("copy tool values buffer", sizeof(bool) * uint64(size) * uint64(size) * uint64(size), EMemoryType::CPU))
#if BENCHMARK
		, colors(Memory::malloc<CompressedColor>("copy tool colors buffer", sizeof(CompressedColor) * uint64(size) * uint64(size) * uint64(size), EMemoryType::CPU))
#endif
	{
        EDIT_TIMES(printf("Size: %u; Allocating %f MB\n", size, Utils::to_MB(uint64(size) * uint64(size) * uint64(size) * (sizeof(uint8) + sizeof(CompressedColor)))));

        EDIT_TIMES(Stats stats);

        EDIT_TIMES(stats.start_work("copying values"));
        DAGUtils::get_values<5>(
                dag,
                values,
                intCopyMin,
                make_uint3(size));

#if COUNT_COPIED_VOXELS
        {
            uint64 num = 0;
            for (uint32 x = 0; x < size; x++)
            {
                for (uint32 y = 0; y < size; y++)
                {
                    for (uint32 z = 0; z < size; z++)
                    {
                        const uint64 index = get_index(x, y, z);
                        if (values[index])
                        {
                            num++;
                        }
                    }
                }
            }
            statsRecorder.report("copied voxels", num);
        }
#endif

        // In benchmark, copy colors before editing to have clean edit stats
#if BENCHMARK && EDITS_ENABLE_COLORS
		PROFILE_SCOPE("Copying colors");
        EDIT_TIMES(stats.start_work("copying colors"));
        for(uint32 x = 0; x < size; x++)
        {
            for (uint32 y = 0; y < size; y++)
            {
                for (uint32 z = 0; z < size; z++)
                {
                    const uint64 index = get_index(x, y, z);
                    if (values[index])
                    {
                        colors[index] = get_color(x, y, z);
                    }
                }
            }
        }
#endif
    }
    ~CopyEditorWithDecompression()
    {
        Memory::free(values);
#if BENCHMARK
        Memory::free(colors);
#endif
    }

    inline CompressedColor get_color(uint32 x, uint32 y, uint32 z) const
    {
	    return DAGUtils::get_color(dagCopy, dagColors, Path(intCopyMin.x + x, intCopyMin.y + y, intCopyMin.z + z));
    }

	inline bool is_valid(uint32 x, uint32 y, uint32 z) const
	{
		return x < size && y < size && z < size;
	}
	inline uint32 is_valid(uint3 p) const
	{
		return is_valid(p.x, p.y, p.z);
	}

	inline uint64 get_index(uint32 x, uint32 y, uint32 z) const
	{
		check(is_valid(x, y, z));
		return uint64(x) + uint64(size) * uint64(y) + uint64(size) * uint64(size) * uint64(z);
	}
	inline uint64 get_index(uint3 p) const
	{
		return get_index(p.x, p.y, p.z);
	}

#if COPY_EMPTY_CHECKS
    inline bool should_edit(const Path path, uint32 depth) const
    {
        if (!Editor<CopyEditorWithDecompression>::should_edit(path, depth))
        {
            return false;
        }
        const uint32 level = C_maxNumberOfLevels - depth;
        if (level < C_maxNumberOfLevels - 4)
        {
            const float3 start = path.as_position(depth);
            const float3 offset = start - dest + source;
            if (offset.x >= 0 && offset.y >= 0 && offset.z >= 0)
            {
                const uint3 position = truncate(offset);
                if (DAGUtils::is_empty(dagCopy, level, position, make_uint3(1u << depth)))
                {
                    return false;
                }
            }
        }
        return true;
    }

	inline uint32 edit_count() const
	{
		return BoxEditorBase::edit_count_impl();
	}
#endif

    inline bool get_new_value(float3 position, bool oldValue) const
    {
        if (oldValue)
        {
            return oldValue;
        }
        else
        {
            const uint3 p = truncate(position - dest + source - copyMin);
            if (!is_valid(p)) return oldValue;
            return values[get_index(p)];
        }
    }
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		if (oldValue == newValue || !newValue)
		{
			return oldColor();
		}
		else
        {
            const uint3 p = truncate(position - dest + source - copyMin);
            if (!is_valid(p)) return oldColor();
#if BENCHMARK
            return colors[get_index(p)];
#else
            return get_color(p.x, p.y, p.z);
#endif
        }
	}
};

#if COPY_WITHOUT_DECOMPRESSION
using CopyEditor = CopyEditorWithoutDecompression<COPY_APPLY_TRANSFORM>;
#else
using CopyEditor = CopyEditorWithDecompression;
#endif

struct FillEditorBase : BufferEditor
{
    uint3 minEdit{};
    uint3 maxEdit{};

	FillEditorBase(const HashDAG& dag, float3 center, float radius)
		: BufferEditor(dag, center, radius)
	{
		Stats stats;
		stats.start_work("flood fill values");

		std::vector<uint3> stack;
		stack.reserve(size * size * size);

		const auto editCenter = make_uint3(size / 2, size / 2, size / 2);
		stack.push_back(editCenter);

		minEdit = editCenter;
		maxEdit = minEdit;

		while (!stack.empty())
		{
			const uint3 position = stack.back();
			stack.pop_back();

			set_value(position, true);
			minEdit = ::min(minEdit, position);
			maxEdit = ::max(maxEdit, position);

			const auto apply_neighbor = [&](auto i, auto j, auto k)
			{
				const auto neighbor = make_uint3(position.x + i, position.y + j, position.z + k);
				if (is_valid(neighbor) && !get_value(neighbor))
				{
					stack.push_back(neighbor);
				}
			};
			apply_neighbor(-1, +0, +0);
			apply_neighbor(+1, +0, +0);
			apply_neighbor(+0, -1, +0);
			apply_neighbor(+0, +1, +0);
			apply_neighbor(+0, +0, -1);
			apply_neighbor(+0, +0, +1);
		}

        boundsMin = min + make_float3(minEdit);
        boundsMax = min + make_float3(maxEdit);
	}
};

struct FillEditorNoColors final : FillEditorBase, Editor<FillEditorNoColors>
{
    using FillEditorBase::FillEditorBase;

	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		if (newValue && !oldValue)
		{
			CompressedColor color;
			color.set_min_color(make_float3(.0f));
			color.set_max_color(make_float3(.5f));
			color.set_weight(std::abs(std::fmod(position.y / 4 - ((int(position.x) % 2) ^ (int(position.z))) / 3.f, 1.f)));
			return color;
		}
		else
		{
			return oldColor();
		}
	}
};

struct FillEditorColors final : FillEditorBase, Editor<FillEditorColors>
{
    CompressedColor* __restrict__ const colors;

	FillEditorColors(const HashDAG& dag, HashDAGColors& dagColors, float3 center, float radius)
		: FillEditorBase(dag, center, radius)
		, colors(Memory::malloc<CompressedColor>("fill editor colors", size * size * size * sizeof(CompressedColor), EMemoryType::CPU))
	{
		Stats stats;
		stats.start_work("flood fill colors");

        CompressedColor color{};
        for (uint32 x = minEdit.x; x <= maxEdit.x; x++)
        {
            for (uint32 y = minEdit.y; y <= maxEdit.y; y++)
            {
                for (uint32 z = minEdit.z; z <= maxEdit.z; z++)
                {
                    const uint64 index = get_index(x, y, z);
                    const uint8 value = values[index];
                    switch (value)
                    {
                        case EValues::Unknown:
                            continue;
                        case EValues::LoadedTrue:
                            color = DAGUtils::get_color(dag, dagColors, get_world_path(x, y, z));
                            break;
                        case EValues::LoadedFalse:
                            continue;
                        case EValues::SetTrue:
                            colors[index] = color;
                            break;
                        case EValues::SetFalse:
                            continue;
                        default:
                            check(false);
                    }
                }
            }
        }
    }
	~FillEditorColors()
    {
	    Memory::free(colors);
    }

	template<typename T>
    inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
    {
        if (newValue && !oldValue)
        {
            const float3 fp = position - min;
            const uint3 p = truncate(fp);
            if (is_valid(p))
            {
                const uint64 index = get_index(p);
                if (values[index] == EValues::SetTrue)
                {
                    return colors[index];
                }
            }
        }
        return oldColor();
    }
};

struct SphereNoiseEditor : BufferEditor, Editor<SphereNoiseEditor>
{
	const float3 center;

	SphereNoiseEditor(const HashDAG& dag, float3 center, float radius, bool isAdding)
		: BufferEditor(dag, center, 1.5f * radius)
		, center(center)
	{
		Stats stats;
		stats.start_work("computing noise");

		const auto editCenter = make_uint3(size / 2, size / 2, size / 2);

		FastNoise noise;
		noise.SetSeed(std::abs(1923 + int32(center.x) + 23 * int32(center.y) + 2938 * int32(center.z)));
		noise.SetFrequency(1 / radius);
		noise.SetFractalOctaves(3);
		for (uint32 x = 0; x < size; x++)
		{
			for (uint32 y = 0; y < size; y++)
			{
				for (uint32 z = 0; z < size; z++)
				{
					const auto position = make_uint3(x, y, z);
					const auto distance = length(editCenter - position);
					if (distance < radius * (1. + noise.GetPerlinFractal(float(x), float(y), float(z))))
					{
						set_value(position, isAdding);
					}
				}
			}
		}
	}
	template<typename T>
	inline CompressedColor get_new_color(float3 position, T oldColor, bool oldValue, bool newValue) const
	{
		if (oldValue == newValue)
		{
			return oldColor();
		}
		CompressedColor color;
		color.bitsPerWeight = 3;
		color.set_min_color(make_float3(94, 30, 0) / 255.f);
		color.set_max_color(make_float3(125, 100, 89) / 255.f);
		color.set_weight(length(position - center) / radius - 0.3f);
		return color;
	}
};
