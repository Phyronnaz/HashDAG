#pragma once

#include "typedefs.h"
#include "utils.h"
#include "array.h"
#include "dags/base_dag.h"
#include "variable_weight_size_colors.h"

struct DAGInfo;

struct BasicDAG : BaseDAG
{
	StaticArray<uint32> data;

	HOST_DEVICE bool is_valid() const
	{
		return data.is_valid();
	}

	HOST_DEVICE uint32 get_first_node_index() const
	{
		return 0;
	}
	HOST_DEVICE uint32 get_node(uint32 level, uint32 index) const
	{
		return data[index];
	}
	HOST_DEVICE uint32 get_child_index(uint32 level, uint32 index, uint8 childMask, uint8 child) const
	{
		return data[index + Utils::child_offset(childMask, child)];
	}
	HOST_DEVICE Leaf get_leaf(uint32 index) const
	{
		return { data[index], data[index + 1] };
	}

	HOST void print_stats() const
	{
		printf("Geometry data: %fMB\n", data.size_in_MB());
	}
	HOST void free()
    {
		data.free();
    }
};

struct BasicDAGColorsBase : BaseDAGColors
{
	// Node count is stored into enclosedLeaves
	uint32 topLevels = 0;
	StaticArray<EnclosedLeavesType> enclosedLeaves;

	HOST_DEVICE uint32 get_color_tree_levels() const
	{
		return 0;
	}
	HOST_DEVICE uint32 get_child_index(uint32 level, uint32 index, uint8 child) const
	{
		check(false);
		return 0;
	}
	HOST_DEVICE uint64 get_leaves_count(uint32 level, uint32 node) const
	{
		const uint32 upperBits = node >> 8;

		// If we are in the top-levels, the obtained value is an index into an 
		// external array
		if (level < topLevels)
		{
			return enclosedLeaves[upperBits];
		}
		else
		{
			return upperBits;
		}
	}
	HOST void print_stats() const
	{
		printf("Enclosed leaves usage:%fMB", enclosedLeaves.size_in_MB());
		printf("##############################################\n");
	}
	HOST void free()
    {
		enclosedLeaves.free();
    }
    HOST void check_ready_for_rt() const
    {
    }
};

struct BasicDAGCompressedColors : BasicDAGColorsBase
{
	using ColorLeaf = CompressedColorLeaf;

	ColorLeaf leaf;

	HOST_DEVICE bool is_valid() const
	{
		return leaf.is_valid();
	}
	HOST_DEVICE ColorLeaf get_leaf(uint32 index) const
	{
		check(false);
		return {};
	}
	HOST_DEVICE ColorLeaf get_default_leaf() const
	{
		return leaf;
	}
	HOST void print_stats() const
	{
		leaf.print_stats();
		BasicDAGColorsBase::print_stats();
	}
	HOST void free()
    {
		BasicDAGColorsBase::free();
	    leaf.free();
    }
};

struct BasicDAGUncompressedColors : BasicDAGColorsBase
{
	struct ColorLeaf
	{
		StaticArray<uint32> colors;

		HOST_DEVICE UncompressedColor get_color(uint32 leaveCount) const
		{
			return { colors[leaveCount] };
		}
		HOST_DEVICE bool is_valid() const
		{
			return colors.is_valid();
		}
		HOST_DEVICE bool is_valid_index(uint32 index) const
		{
			return colors.is_valid_index(index);
        }
        HOST void free()
        {
			colors.free();
        }
	};

	ColorLeaf leaf;

	HOST_DEVICE bool is_valid() const
	{
		return leaf.is_valid();
	}
	HOST_DEVICE ColorLeaf get_leaf(uint32 index) const
	{
		check(false);
		return {};
	}
	HOST_DEVICE ColorLeaf get_default_leaf() const
	{
		return leaf;
	}
	HOST void print_stats() const
	{
		printf(
			"Color stats:\n"
			"\t%" PRIu64 " colors\n"
			"\t%fMB used by colors\n",
			leaf.colors.size(),
			leaf.colors.size_in_MB());
		printf("##############################################\n");
		BasicDAGColorsBase::print_stats();
	}
	HOST void free()
    {
		BasicDAGColorsBase::free();
	    leaf.free();
    }
};

struct BasicDAGColorErrors : BaseDAGColors
{
	struct ColorLeaf
	{
		CompressedColorLeaf compressedColorLeaf;
		BasicDAGUncompressedColors::ColorLeaf uncompressedColorLeaf;

		HOST_DEVICE UncompressedColor get_color(uint32 leaveCount) const
		{
			const CompressedColor compressed = compressedColorLeaf.get_color(leaveCount);
			const UncompressedColor uncompressed = uncompressedColorLeaf.get_color(leaveCount);

			return ColorUtils::color_error(compressed.get_color(), uncompressed.get_color()) > 0.04 ? 1.f : 0.f;
		}
		HOST_DEVICE bool is_valid() const
		{
			return compressedColorLeaf.is_valid() && uncompressedColorLeaf.is_valid();
		}
		HOST_DEVICE bool is_valid_index(uint32 index) const
		{
			return compressedColorLeaf.is_valid_index(index) && uncompressedColorLeaf.is_valid_index(index);
		}
	};

	ColorLeaf leaf;

	BasicDAGCompressedColors compressedColors;
	BasicDAGUncompressedColors uncompressedColors;

	HOST_DEVICE bool is_valid() const
	{
		return compressedColors.is_valid() && uncompressedColors.is_valid();
	}

	HOST_DEVICE uint32 get_color_tree_levels() const
	{
		return 0;
	}
	HOST_DEVICE uint32 get_child_index(uint32 level, uint32 index, uint8 child) const
	{
		check(false);
		return 0;
	}
	HOST_DEVICE uint64 get_leaves_count(uint32 level, uint32 node) const
	{
		return compressedColors.get_leaves_count(level, node);
	}
	HOST_DEVICE ColorLeaf get_leaf(uint32 index) const
	{
		check(false);
		return {};
	}
	HOST_DEVICE ColorLeaf get_default_leaf() const
	{
		return { compressedColors.get_default_leaf(), uncompressedColors.get_default_leaf() };
	}
	HOST void free()
    {
	    // Nothing to do, only refs we own nothing
    }
    HOST void check_ready_for_rt() const
    {
    }
};

struct BasicDAGFactory
{
	static void save_uncompressed_colors_to_file(const BasicDAGUncompressedColors& colors, const std::string& path);
	static void load_uncompressed_colors_from_file(BasicDAGUncompressedColors& outColors, const std::string& path);

	static void save_compressed_colors_to_file(const BasicDAGCompressedColors& colors, const std::string& path);
    static void load_compressed_colors_from_file(BasicDAGCompressedColors& outColors, const std::string& path, bool enclosedLeavesCompat32 = false);

	static void save_dag_to_file(const DAGInfo& info, const BasicDAG& dag, const std::string& path);
	static void load_dag_from_file(DAGInfo& outInfo, BasicDAG& outDag, const std::string& path);
};