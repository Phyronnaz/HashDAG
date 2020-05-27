#pragma once

#include "typedefs.h"
#include "stats.h"
#include "variable_weight_size_colors.h"
#include "dags/base_dag.h"
#include "hash_dag_globals.h"

struct HashDAGColors : BaseDAGColors
{
	using ColorLeaf = CompressedColorLeaf;

public:
#ifdef __CUDA_ARCH__
#define nodes nodes_GPU
#define leaves leaves_GPU
#define offsets offsets_GPU
#else
#define nodes nodes_CPU
#define leaves leaves_CPU
#define offsets offsets_CPU
#endif

	HOST_DEVICE uint32 get_color_tree_levels() const
	{
		return C_colorTreeLevels;
	}
	HOST_DEVICE static uint64 get_leaves_count(uint32 level, uint32 node)
	{
		check(level >= C_colorTreeLevels);
		return node >> 8;
	}
	HOST_DEVICE uint32 get_child_index(uint32 level, uint32 index, uint8 child) const
	{
		return nodes[index + child];
	}
	HOST_DEVICE ColorLeaf get_leaf(uint32 index) const
	{
        if (Utils::has_flag(index))
        {
            return leaves[Utils::clear_flag(index)];
        }
        else
        {
            const uint64 offset = offsets[index];
            ColorLeaf leaf = mainLeaf;
            leaf.set_as_shared(offset);
            return leaf;
        }
	}
	HOST_DEVICE ColorLeaf get_default_leaf() const
	{
		return {};
	}

	HOST_DEVICE bool is_valid() const
	{
		return nodes.is_valid();
	}

#undef nodes
#undef leaves
#undef offsets

private:
	DynamicArray<uint32> nodes_GPU; // Leaves are indices to leaves if top bit is 1, else offsets
	DynamicArray<ColorLeaf> leaves_GPU;
	DynamicArray<uint64> offsets_GPU;
	ColorLeaf mainLeaf;

	DynamicArray<uint32> nodes_CPU;
	DynamicArray<ColorLeaf> leaves_CPU;
	DynamicArray<uint64> offsets_CPU;

public:
    HOST uint32 allocate_interior_node()
    {
        return cast<uint32>(nodes_CPU.add(0));
    }
    HOST uint32 allocate_leaf()
    {
        CompressedColorLeaf leaf;
        leaf.set_as_unique();
        return Utils::set_flag(cast<uint32>(leaves_CPU.add(leaf)));
    }
    HOST uint32 get_node(uint32 index)
    {
        check(!Utils::has_flag(index));
        return nodes_CPU[index];
    }
    HOST void set_node(uint32 index, uint32 value)
    {
        nodes_CPU[index] = value;
    }

    HOST ColorLeaf& get_leaf_ref(uint32 leafIndex)
    {
        check(Utils::has_flag(leafIndex));
        return leaves_CPU[Utils::clear_flag(leafIndex)];
    }

    HOST void upload_to_gpu(bool firstUpload)
    {
        PROFILE_FUNCTION();
    	
        nodes_CPU.copy_to_gpu_flexible(nodes_GPU);
        leaves_CPU.copy_to_gpu_flexible(leaves_GPU);
        if (firstUpload)
        {
            // offsets are never edited
            offsets_CPU.copy_to_gpu_flexible(offsets_GPU);
        }
    }
    HOST void upload_leaf_index_to_gpu_async(uint32 leafIndex)
    {
        check(Utils::has_flag(leafIndex));
        check(leaves_CPU.size() == leaves_GPU.size());
        leafIndex = Utils::clear_flag(leafIndex);
        CUDA_CHECKED_CALL cudaMemcpyAsync(&leaves_GPU[leafIndex], &leaves_CPU[leafIndex], sizeof(ColorLeaf), cudaMemcpyHostToDevice);
    }

    HOST void free()
    {
        for (auto& leaf : leaves_CPU)
        {
            leaf.free();
        }
        nodes_CPU.free();
        nodes_GPU.free();
        leaves_CPU.free();
        leaves_GPU.free();
        offsets_CPU.free();
        offsets_GPU.free();
    }

    HOST void check_ready_for_rt() const
    {
        check(nodes_CPU.size() == nodes_GPU.size());
        check(leaves_CPU.size() == leaves_GPU.size());
        check(offsets_CPU.size() == offsets_GPU.size());
    }

    HOST double get_total_used_memory() const
    {
        double usage = 0;
        for (auto& leaf : leaves_CPU)
        {
            // TODO check(!leaf.is_shared());
            usage += leaf.blocks_CPU.size_in_MB();
            usage += leaf.weights_CPU.size_in_MB();
            usage += leaf.macroBlocks_CPU.size_in_MB();
        }
        usage += nodes_CPU.allocated_size_in_MB();
        usage += leaves_CPU.allocated_size_in_MB();
        usage += offsets_CPU.allocated_size_in_MB();
        usage += mainLeaf.size_in_MB();
        return usage;
    }

	HOST void print_stats() const
	{
		double blocksSizeInMB = 0;
		double weightsSizeInMB = 0;
		double macroBlocksSizeInMB = 0;
		for (auto& leaf : leaves_CPU)
        {
            check(!leaf.is_shared());
            blocksSizeInMB += leaf.blocks_CPU.size_in_MB();
            weightsSizeInMB += leaf.weights_CPU.size_in_MB();
            macroBlocksSizeInMB += leaf.macroBlocks_CPU.size_in_MB();
        }
		printf(
			"Color stats, ignoring the main leaf:\n"
			"\t%" PRIu64 " color tree nodes\n"
			"\t%" PRIu64 " color tree leaves\n"
			"\t%" PRIu64 " color tree offsets\n"
			"\t%fMB used by tree nodes (%fMB allocated)\n"
			"\t%fMB used by tree leaves (%fMB allocated)\n"
			"\t%fMB used by tree offsets (%fMB allocated)\n"
			"\t%fMB used by blocks\n"
			"\t%fMB used by weights\n"
			"\t%fMB used by macro blocks\n"
			"\tTotal: %fMB\n",
			nodes_CPU.size(),
			leaves_CPU.size(),
			offsets_CPU.size(),
			nodes_CPU.size_in_MB(), nodes_CPU.allocated_size_in_MB(),
			leaves_CPU.size_in_MB(), leaves_CPU.allocated_size_in_MB(),
			offsets_CPU.size_in_MB(), offsets_CPU.allocated_size_in_MB(),
			blocksSizeInMB,
			weightsSizeInMB,
			macroBlocksSizeInMB,
			nodes_CPU.allocated_size_in_MB() + leaves_CPU.allocated_size_in_MB() + offsets_CPU.allocated_size_in_MB() + blocksSizeInMB + weightsSizeInMB);
		printf("##############################################\n");
	}

	friend struct HashColorsBuilder;
};

struct HashColorsBuilder
{
	struct BuildLeaf
	{
		explicit BuildLeaf(uint64 offset)
			: offset(offset)
		{
		}

		uint64 offset = 0; // offset into the global shared leaf
		ColorLeafBuilder builder;
	};
	std::vector<uint32> nodes;
	std::vector<BuildLeaf> leaves;

	HOST void build(HashDAGColors& tree, const CompressedColorLeaf& globalLeaf) const
	{
        PROFILE_FUNCTION();
		
	    checkAlways(!tree.is_valid());
	    tree.mainLeaf = globalLeaf;

		printf("\t%" PRIu64 " color nodes for the upper levels tree\n", uint64(nodes.size()));
		printf("\t%" PRIu64 " leaves\n", uint64(leaves.size()));

		Stats stats;
		stats.start_work("building nodes");
		tree.nodes_CPU = DynamicArray<uint32>::allocate("hash colors nodes", nodes.size(), EMemoryType::CPU);
		for (uint64 index = 0; index < nodes.size(); index++)
		{
            tree.nodes_CPU[index] = nodes[index];
            check(nodes[index] < max(nodes.size(), leaves.size()));
        }
        stats.start_work("building leaves");

        tree.leaves_CPU = DynamicArray<CompressedColorLeaf>::allocate("hash colors leaves", 1, EMemoryType::CPU);
        tree.offsets_CPU = DynamicArray<uint64>::allocate("hash colors offsets", leaves.size(), EMemoryType::CPU);

        // can't allocate an array of size 0
        tree.leaves_CPU.hack_set_size(0);

        uint32 offsetsCounter = 0;
        for (auto& leaf : leaves)
        {
            tree.offsets_CPU[offsetsCounter++] = leaf.offset;
        }

        tree.upload_to_gpu(true);

#if BENCHMARK
		// Reserve more space to avoid reallocs when benchmarking
		tree.leaves_CPU.reserve(128 * 1024);
		tree.nodes_CPU.reserve(tree.nodes_CPU.size());
		tree.leaves_GPU.reserve(128 * 1024);
		tree.nodes_GPU.reserve(tree.nodes_GPU.size());
#endif
	}
};