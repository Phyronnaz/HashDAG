#pragma once

#include "typedefs.h"
#include "color_utils.h"

struct Leaf
{
	uint32 low;
	uint32 high;

	HOST_DEVICE uint64 to_64() const
	{
		return (uint64(high) << 32) | uint64(low);
	}

	HOST_DEVICE uint8 get_first_child_mask() const
	{
#if defined(__CUDA_ARCH__)
		const uint32 u0 = low | (low >> 1);
		const uint32 u1 = u0 | (u0 >> 2);
		const uint32 u2 = u1 | (u1 >> 4);
		const uint32 t0 = u2 & 0x01010101u;

		const uint32 v0 = high | (high >> 1);
		const uint32 v1 = v0 | (v0 >> 2);
		const uint32 v2 = v1 | (v1 << 4); // put into second nibble
		const uint32 s0 = v2 & 0x10101010u;

		uint32 s1;
		asm(
			"{\n\t"
			"    .reg .u32 t0;\n\t"
			"    mul.lo.u32 t0, %1, 0x1020408;\n\t" // multiply keeping the low part
			"    mad.lo.u32 %0, %2, 0x1020408, t0;\n\t" // multiply keeping the low part and add
			"}\n\t"
			: "=r"(s1) : "r"(t0), "r"(s0)
		);

		return s1 >> 24;
#else
		const uint64 currentLeafMask = to_64();
		return uint8(
			((currentLeafMask & 0x00000000000000FF) == 0 ? 0 : 1 << 0) |
			((currentLeafMask & 0x000000000000FF00) == 0 ? 0 : 1 << 1) |
			((currentLeafMask & 0x0000000000FF0000) == 0 ? 0 : 1 << 2) |
			((currentLeafMask & 0x00000000FF000000) == 0 ? 0 : 1 << 3) |
			((currentLeafMask & 0x000000FF00000000) == 0 ? 0 : 1 << 4) |
			((currentLeafMask & 0x0000FF0000000000) == 0 ? 0 : 1 << 5) |
			((currentLeafMask & 0x00FF000000000000) == 0 ? 0 : 1 << 6) |
			((currentLeafMask & 0xFF00000000000000) == 0 ? 0 : 1 << 7));
#endif
	}
	HOST_DEVICE uint8 get_second_child_mask(uint8 firstChildIndex)
	{
		const uint8 shift = uint8((firstChildIndex & 3) * 8);
		const uint32 leafMask = (firstChildIndex & 4) ? high : low;
		return uint8(leafMask >> shift);
	}
};

struct BaseDAG
{
    constexpr static uint32 levels = MAX_LEVELS;

	HOST_DEVICE constexpr uint32 leaf_level() const
	{
		return levels - 2;
	}
	HOST_DEVICE constexpr bool is_leaf(uint32 level) const
	{
		return level == leaf_level();
	}

#ifdef __PARSER__
	uint32 get_first_node_index() const;
	uint32 get_node(uint32 level, uint32 index) const;
	// level: level of index, not of the child!
	uint32 get_child_index(uint32 level, uint32 index, uint8 childMask, uint8 child) const;
	Leaf get_leaf(uint32 index) const;
#endif
};

struct BaseDAGColors
{
#ifdef __PARSER__
	using ColorLeaf = CompressedColorLeaf;

	// Number of levels to go down before counting leaves
	uint32 get_color_tree_levels() const;

	// Number of leaves this node has
	uint64 get_leaves_count(uint32 level, uint32 node) const;

	uint32 get_child_index(uint32 level, uint32 index, uint8 child) const;
	ColorLeaf get_leaf(uint32 index) const;
	ColorLeaf get_default_leaf() const;
#endif
};