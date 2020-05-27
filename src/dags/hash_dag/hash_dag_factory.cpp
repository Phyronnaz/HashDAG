#include "hash_dag_factory.h"
#include "dag_tracer.h"
#include "serializer.h"

uint32 create_hash_dag(
	const BasicDAG& sdag,
	HashDAG& hdag,
	std::vector<uint32>& map,
	const uint32 level,
	const uint32 index)
{
	const bool isLeaf = sdag.is_leaf(level);
	uint32 finalIndex;
	if (!isLeaf)
	{
		const uint32 node = sdag.get_node(level, index);
		const uint8 childMask = Utils::child_mask(node);

        uint32 nodeBuffer[9];
        uint32 nodeBufferSize = 0;
        nodeBuffer[nodeBufferSize++] = node;

		for (uint8 i = 0; i < 8; ++i)
		{
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				uint32 newChildIndex = map[childIndex];
				if (newChildIndex == 0)
				{
					newChildIndex = create_hash_dag(sdag, hdag, map, level + 1, childIndex);
					map[childIndex] = newChildIndex + 1;
				}
				else
				{
					newChildIndex--;
				}
				nodeBuffer[nodeBufferSize++] = newChildIndex;
			}
		}

		const uint32 hash = HashDagUtils::hash_interior(nodeBufferSize, nodeBuffer);

#if USE_BLOOM_FILTER
        BloomFilter filter;
        hdag.data.bloom_filter_init_interior(filter, nodeBufferSize, nodeBuffer);
#endif // ~ USE_BLOOM_FILTER

        finalIndex = hdag.data.add_interior_node(level, nodeBufferSize, nodeBuffer, hash BLOOM_FILTER_ARG(filter));
	}
	else
	{
		const uint64 leaf = sdag.get_leaf(index).to_64();
		const uint32 hash = HashDagUtils::hash_leaf(leaf);

#if USE_BLOOM_FILTER
        BloomFilter filter;
        hdag.data.bloom_filter_init_leaf(filter, leaf);
#endif // ~ USE_BLOOM_FILTER

        finalIndex = hdag.data.add_leaf_node(level, leaf, hash BLOOM_FILTER_ARG(filter));
    }

#if 0
    {
        for (int index = 0; index < newNodeData.size(); index++)
        {
            check(newNodeData[index] == hdag.get_node(level, finalIndex + index));
        }
        auto checkEqual = [&](int level, uint32 oldIndex, uint32 newIndex)
        {
            uint32 oldNodeTemp = sdag.get_node(level, oldIndex);
            uint32 newNodeTemp = hdag.get_node(level, newIndex);
            check(oldNodeTemp == newNodeTemp);
        };
        if (isLeaf)
        {
            checkEqual(level, index.index, finalIndex);
            checkEqual(level, index.index + 1, finalIndex + 1);
        }
        else
        {
            checkEqual(level, index.index, finalIndex);
            const uint32 node = sdag.get_node(level, index);
            const uint8 childMask = Utils::child_mask(node);
            for (uint8 i = 0; i < 8; ++i)
            {
                if (childMask & (1u << i))
                {
                    uint32 oldChildIndex = sdag.get_child_index(level, index, childMask, i);
                    uint32 newChildIndex = hdag.get_child_index(level, finalIndex, childMask, i);
                    checkEqual(level + 1, oldChildIndex.index, newChildIndex.index);
                }
            }
        }
    }
#endif
	return finalIndex;
}

uint32 create_hash_dag_colors(
	const BasicDAG& sdag,
	const BasicDAGCompressedColors& sdagcolors,
	HashColorsBuilder& colorBuilder,
	const uint32 level,
	const uint32 index,
	uint64 leavesCount)
{
	const uint32 node = sdag.get_node(level, index);
	const uint8 childMask = Utils::child_mask(node);
	const uint32 colorIndex = (uint32)colorBuilder.nodes.size();

	check(C_colorTreeLevels < sdag.leaf_level());

	if (level == C_colorTreeLevels - 1)
	{
		for (uint8 i = 0; i < 8; i++)
		{
            colorBuilder.nodes.push_back((uint32)colorBuilder.leaves.size());
            colorBuilder.leaves.emplace_back(HashColorsBuilder::BuildLeaf{ leavesCount });
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				const uint32 childNode = sdag.get_node(level + 1, childIndex);
				leavesCount += sdagcolors.get_leaves_count(level + 1, childNode);
			}
		}
	}
	else
	{
		for (uint8 i = 0; i < 8; i++)
		{
			colorBuilder.nodes.push_back(0);
		}
		for (uint8 i = 0; i < 8; i++)
		{
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				const uint32 childColorIndex = create_hash_dag_colors(sdag, sdagcolors, colorBuilder, level + 1, childIndex, leavesCount);
				check(colorBuilder.nodes[colorIndex + i] == 0);
				colorBuilder.nodes[colorIndex + i] = childColorIndex;

				const uint32 childNode = sdag.get_node(level + 1, childIndex);
				leavesCount += sdagcolors.get_leaves_count(level + 1, childNode);
			}
		}
	}
	return colorIndex;
}

void HashDAGFactory::load_from_DAG(HashDAG& outDag, const BasicDAG& inDag, uint32 numPages)
{
	PROFILE_FUNCTION();
	SCOPED_STATS("Creating hash dag");

	Stats stats;

	stats.start_work("Allocating pool");
	outDag.data.create(numPages);

#if ADD_FULL_NODES_FIRST
	stats.start_work("Adding full nodes");
    outDag.data.cpuData.fullNodeIndices = new uint32[MAX_LEVELS];
    for (uint32 level = inDag.leaf_level(); level > 0; level--)
    {
        outDag.data.add_full_node(level);
    }
#endif
	
	stats.start_work("Hashing existing dag");
	std::vector<uint32> map(inDag.data.size(), 0);
	outDag.firstNodeIndex = create_hash_dag(inDag, outDag, map, 0, 0);

	stats.start_work("Checking");
	// outDag.check_nodes();

#if !MANUAL_VIRTUAL_MEMORY
	outDag.pool = outDag.data.gpuPool;
#endif

    stats.start_work("upload_to_gpu");
    outDag.data.upload_to_gpu();
}

void HashDAGFactory::load_colors_from_DAG(
	HashDAGColors& outDagColors,
	const BasicDAG& inDag, 
	const BasicDAGCompressedColors& inDagColors)
{
	PROFILE_FUNCTION();
	SCOPED_STATS("Creating hash dag colors");
	
	HashColorsBuilder colorBuilder;
	const uint32 colorIndex =  create_hash_dag_colors(inDag, inDagColors, colorBuilder, 0, 0, 0);
	checkAlways(colorIndex == 0);
	colorBuilder.build(outDagColors, inDagColors.leaf);
}

void HashDAGFactory::save_dag_to_file(const DAGInfo& info, const HashDAG& dag, const std::string& path)
{
	PROFILE_FUNCTION();
	checkAlways(dag.is_valid());
	
	FileWriter writer(path);
	
	writer.write(info);
	writer.write(dag.levels);

	writer.write(dag.firstNodeIndex);
	writer.write(dag.data.cpuData.poolMaxSize);
	writer.write(dag.data.pageTableSize);
	writer.write(dag.data.poolTop);

#if MANUAL_CPU_DATA
	writer.write(dag.data.cpuData.cpuPool, dag.data.poolTop * C_pageSize * sizeof(uint32));
	writer.write(dag.data.cpuData.cpuPageTable, dag.data.pageTableSize * sizeof(uint32));
#else
	checkAlways(false);
#endif
}

void HashDAGFactory::load_dag_from_file(DAGInfo& info, HashDAG& dag, const std::string& path)
{
	PROFILE_FUNCTION();
	checkAlways(!dag.is_valid());
	
	FileReader reader(path);
	
	reader.read(info);
	uint32 levels = 0;
	reader.read(levels);
    checkfAlways(levels == MAX_LEVELS, "MAX_LEVELS is %u, should be %u", MAX_LEVELS, levels);

	reader.read(dag.firstNodeIndex);
	reader.read(dag.data.cpuData.poolMaxSize);
	reader.read(dag.data.pageTableSize);
	reader.read(dag.data.poolTop);

#if MANUAL_CPU_DATA
	reader.read(dag.data.cpuData.cpuPool, dag.data.poolTop * C_pageSize * sizeof(uint32));
	reader.read(dag.data.cpuData.cpuPageTable, dag.data.pageTableSize * sizeof(uint32));
#else
	checkAlways(false);
#endif
}