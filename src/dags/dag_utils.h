#pragma once

#include "typedefs.h"
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace DAGUtils
{
    template<typename TDAG>
    inline void print_stats(const TDAG& dag)
    {
        printf("##############################################\n");
        std::vector<std::unordered_map<uint32, uint32>> multipliers(MAX_LEVELS);

        struct ChildrenCount
        {
            uint32 counts[8] = {};
        };

        std::vector<uint64> nodeCountsDAG(MAX_LEVELS, 0);
        std::vector<uint64> nodeCountsSVO(MAX_LEVELS, 0);
        std::vector<uint64> realChildren(MAX_LEVELS, 0);
        std::vector<ChildrenCount> realChildren2(MAX_LEVELS);
        std::vector<uint64> totalChildren(MAX_LEVELS, 0);

        uint64 totalDAGVoxels = 0;
        uint64 totalSVOVoxels = 0;

        multipliers[0].emplace(dag.get_first_node_index(), 1u);

        for (uint32 level = 0; level < dag.levels - 1; level++)
        {
            for (auto it : multipliers[level])
            {
                const uint32 index = it.first;
                const uint32 multiplier = it.second;
                check(multiplier != 0);

                nodeCountsDAG[level]++;
                nodeCountsSVO[level] += multiplier;

                if (level < dag.leaf_level())
                {
                    const uint32 node = dag.get_node(level, index);
                    const uint8 childMask = Utils::child_mask(node);
                    realChildren2[level].counts[Utils::popc(childMask) - 1]++;
                    for (uint8 child = 0; child < 8; ++child)
                    {
                        totalChildren[level]++;
                        if (childMask & (1u << child))
                        {
                            realChildren[level]++;
                            uint32 childIndex = dag.get_child_index(level, index, childMask, child);
                            multipliers[level + 1][childIndex] += multiplier;
                        }
                    }
                }
                else
                {
                    const uint32 count = Utils::popcll(dag.get_leaf(index).to_64());
                    totalDAGVoxels += count;
                    totalSVOVoxels += count * multiplier;
                }
            }
        }
        uint64 totalSVO = 0;
        uint64 totalDAG = 0;
        uint64 totalRealChildren = 0;
        uint64 totalTotalChildren = 0;
        for (uint32 level = 0; level < dag.levels - 1; level++)
        {
            uint64 SVO = nodeCountsSVO[level];
            uint64 DAG = nodeCountsDAG[level];
            uint64 realChildrenForLevel = realChildren[level];
            uint64 totalChildrenForLevel = totalChildren[level];
            ChildrenCount totalChildrenForLevel2 = realChildren2[level];

            int64 difference = int64(SVO) - int64(DAG);

            totalSVO += SVO;
            totalDAG += DAG;
            totalRealChildren += realChildrenForLevel;
            totalTotalChildren += totalChildrenForLevel;

            printf(
                    "\nlevel %d:"
                    "\n\tSVO: %" PRIu64
                    "\n\tDAG: %" PRIu64
                    "\n\tdifference: %" PRId64
                    "\n\tcompression: %2.2f%%"
                    "\n\t1: %u"
                    "\n\t2: %u"
                    "\n\t3: %u"
                    "\n\t4: %u"
                    "\n\t5: %u"
                    "\n\t6: %u"
                    "\n\t7: %u"
                    "\n\t8: %u"
                    "\n\tavg children: %2.2f",
                    level,
                    SVO,
                    DAG,
                    difference,
                    SVO == 0 ? 0 : (int(1000. * double(DAG) / double(SVO)) / 10.),
                    totalChildrenForLevel2.counts[0],
                    totalChildrenForLevel2.counts[1],
                    totalChildrenForLevel2.counts[2],
                    totalChildrenForLevel2.counts[3],
                    totalChildrenForLevel2.counts[4],
                    totalChildrenForLevel2.counts[5],
                    totalChildrenForLevel2.counts[6],
                    totalChildrenForLevel2.counts[7],
                    8 * double(realChildrenForLevel) / double(totalChildrenForLevel)
            );
        }
        int64 difference = int64(totalSVO) - int64(totalDAG);
        printf(
                "\ntotal:"
                "\n\tSVO: %" PRIu64
                "\n\tDAG: %" PRIu64
                "\n\tdifference: %" PRId64
                "\n\tcompression: %2.2f%%"
                "\n\tavg children: %2.2f"
                "\n\n",
                totalSVO,
                totalDAG,
                difference,
                totalSVO == 0 ? 0 : (int(1000. * double(totalDAG) / double(totalSVO)) / 10.),
                8 * double(totalRealChildren) / double(totalTotalChildren)
        );
        printf("Total DAG voxels: %" PRIu64 "\n", totalDAGVoxels);
        printf("Total SVO voxels: %" PRIu64 "\n", totalSVOVoxels);
        printf("##############################################\n");
    }

    template<typename TDAG>
    HOST_DEVICE bool get_value(const TDAG& dag, const Path path)
    {
        PROFILE_FUNCTION_SLOW();
    	
        uint32 nodeIndex = dag.get_first_node_index();
        for (uint32 level = 0; level < dag.levels; level++)
        {
            if (level < dag.leaf_level())
            {
                const uint32 node = dag.get_node(level, nodeIndex);
                const uint8 childMask = Utils::child_mask(node);
                const uint8 child = path.child_index(level + 1, dag.levels);
                if (!(childMask & (1u << child)))
                {
                    return false;
                }
                nodeIndex = dag.get_child_index(level, nodeIndex, childMask, child);
            }
            else
            {
                const Leaf leaf = dag.get_leaf(nodeIndex);
                const uint8 leafBitIndex = uint8(
                        (((path.path.x & 0x1) == 0) ? 0 : 4) |
                        (((path.path.y & 0x1) == 0) ? 0 : 2) |
                        (((path.path.z & 0x1) == 0) ? 0 : 1) |
                        (((path.path.x & 0x2) == 0) ? 0 : 32) |
                        (((path.path.y & 0x2) == 0) ? 0 : 16) |
                        (((path.path.z & 0x2) == 0) ? 0 : 8));
                return leaf.to_64() & (uint64(1) << leafBitIndex);
            }
        }
        check(false);
        return true;
    }

    template<typename TDAG>
    HOST_DEVICE_RECURSIVE bool is_empty_impl(
            const TDAG& dag,
            uint32 nodeIndex,
            Path path,
            const uint32 level,
            const uint32 maxLevel,
            const uint3 start,
            const uint3 size)
    {
        if (level == maxLevel)
        {
            return false;
        }

        const auto shouldEdit = [start, size](auto& p, uint32 shift)
        {
            const uint3 boundsMin = p.path << shift;
            const uint3 boundsMax = boundsMin + make_uint3(uint32((1 << shift) - 1)); // Inclusive
            const uint3 inMin = start;
            const uint3 inMax = start + size;
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
        };
        if (level < C_leafLevel && !shouldEdit(path, dag.levels - level))
        {
            return true;
        }

        const uint32* nodePtr = dag.data.get_sys_ptr(level, nodeIndex);
        const uint8 childMask = Utils::child_mask(nodePtr[0]);

        // Iterate children
        uint32 nodeChildOffset = 1;
        for (uint8 child = 0; child < 8; child++)
        {
            Path newPath = path;
            newPath.descend(child);
            if (childMask & (1u << child))
            {
                const uint32 childNodeIndex = nodePtr[nodeChildOffset++];
                const bool empty = is_empty_impl(
                        dag,
                        childNodeIndex,
                        newPath,
                        level + 1,
                        maxLevel,
                        start,
                        size);
                if (!empty)
                {
                    return false;
                }
            }
        }
        return true;
    }
    template<typename TDAG>
    HOST_DEVICE bool is_empty(
            const TDAG& dag,
            const uint32 maxLevel,
            const uint3 start,
            const uint3 size)
    {
        PROFILE_FUNCTION_SLOW();
        checkAlways(maxLevel <= C_leafLevel);
        return is_empty_impl(dag, dag.get_first_node_index(), Path(make_uint3(0)), 0, maxLevel, start, size);
    }

    template<int32 ThreadsHeight, uint32 level, typename TDAG>
    HOST_DEVICE void get_values_impl(
            const TDAG& dag,
            uint32 nodeIndex,
            Path path,
            bool* __restrict__ values,
            const uint3 start,
            const uint3 size,
            std::vector<std::function<void()>>& tasks)
    {
        const auto shouldEdit = [start, size](auto& p, uint32 shift)
        {
            const uint3 boundsMin = p.path << shift;
            const uint3 boundsMax = boundsMin + make_uint3(uint32((1 << shift) - 1)); // Inclusive
            const uint3 inMin = start;
            const uint3 inMax = start + size;
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
        };
        if (level < C_leafLevel && !shouldEdit(path, dag.levels - level))
        {
            return;
        }

#if defined(__CUDACC__) && !defined(__PARSER__) // no C++17 support
        if (true)
#else
        if constexpr (level == C_leafLevel)
#endif
        {
            const uint32* nodePtr = dag.data.get_sys_ptr(level, nodeIndex);
            const uint32 low = nodePtr[0];
            const uint32 high = nodePtr[1];

            // Iterate level 1 children
            for (uint8 child1 = 0; child1 < 8; child1++)
            {
                Path newPath1 = path;
                newPath1.descend(child1);
                if (shouldEdit(newPath1, 1))
                {
                    // Iterate level 2 children
                    for (uint8 child2 = 0; child2 < 8; child2++)
                    {
                        Path newPath2 = newPath1;
                        newPath2.descend(child2);

                        if (shouldEdit(newPath2, 0))
                        {
                            const uint32 leafMask = (child1 & 4) ? high : low;
                            const uint8 bitIndex = uint8((child1 & 3) * 8 + child2);
                            check(bitIndex < 32);
                            const uint32 mask = 1u << bitIndex;
                            const bool value = leafMask & mask;

                            check(start <= newPath2.path);
                            const uint3 position = newPath2.path - start;
                            check(position < size);
                            values[uint64(position.x) + uint64(size.x) * uint64(position.y) + uint64(size.x) * uint64(size.y) * uint64(position.z)] = value;
                        }
                    }
                }
            }
        }
        else
        {
            const uint32* nodePtr = dag.data.get_sys_ptr(level, nodeIndex);
            const uint8 childMask = Utils::child_mask(nodePtr[0]);

            // Iterate children
            uint32 nodeChildOffset = 1;
            for (uint8 child = 0; child < 8; child++)
            {
                Path newPath = path;
                newPath.descend(child);
                if (childMask & (1u << child))
                {
                    const uint32 childNodeIndex = nodePtr[nodeChildOffset++];
                    const auto lambda = [&dag, &tasks, childNodeIndex, newPath, values, start, size]()
                    {
                        get_values_impl<ThreadsHeight, level + 1>(
                                dag,
                                childNodeIndex,
                                newPath,
                                values,
                                start,
                                size,
                                tasks);
                    };
                    if (dag.leaf_level() - level == ThreadsHeight)
                    {
                        tasks.push_back(lambda);
                    }
                    else
                    {
                        lambda();
                    }
                }
            }
        }
    }
    template<int32 ThreadsHeight, typename TDAG>
    HOST_DEVICE void get_values(
            const TDAG& dag,
            bool*__restrict__ values,
            const uint3 start,
            const uint3 size)
    {
        PROFILE_FUNCTION();
        EDIT_TIMES(Stats stats);

        const uint64 num = uint64(size.x) * uint64(size.y) * uint64(size.z);

        EDIT_TIMES(printf("copying %" PRIu64 " voxels\n", num));

        EDIT_TIMES(stats.start_work("memset"));
        // memset now so that we can skip writing when reaching empty nodes
        std::memset(values, 0, num);

        EDIT_TIMES(stats.start_work("copying"));
        std::vector<std::function<void()>> tasks;
        get_values_impl<ThreadsHeight, 0>(dag, dag.get_first_node_index(), Path(0, 0, 0), values, start, size, tasks);
        std::vector<std::thread> threads;
#if NUM_THREADS == 0
        threads.reserve(tasks.size());
        for (auto& task : tasks)
        {
            threads.emplace_back(task);
        }
#else
        std::atomic<uint32> taskCounter{ 0 };
        threads.reserve(NUM_THREADS);
        const auto lambda = [&](uint32 threadIndex)
        {
            for (;;)
            {
                const uint32 taskIndex = taskCounter.fetch_add(1);
                if (taskIndex >= tasks.size()) break;
                tasks[taskIndex]();
                if (taskIndex % 100 == 0)
                {
                    EDIT_TIMES(printf("Task %u/%u    \r", taskIndex, uint32(tasks.size())));
                }
            }
        };
        for (uint32 index = 0; index < NUM_THREADS; index++)
        {
            threads.emplace_back([index, &lambda]() { lambda(index); });
        }
#endif
        EDIT_TIMES(printf("%" PRIu64 " tasks, %" PRIu64 " threads\n", uint64(tasks.size()), uint64(threads.size())));

        for (auto& thread : threads)
        {
            thread.join();
        }
    }

    // Copy paste from tracer.cu
    template<typename TDAG, typename TDAGColors>
    HOST_DEVICE auto get_color(const TDAG& dag, const TDAGColors& colors, const Path path)
    {
        PROFILE_FUNCTION_SLOW();
    	
        uint64 nof_leaves = 0;

        uint32 colorNodeIndex = 0;
        typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();
        using ColorType = decltype(colorLeaf.get_color(0));

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
                return ColorType();
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
                        return ColorType();
                    }
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
                const uint8 leafBitIndex = uint8(
                        (((path.path.x & 0x1) == 0) ? 0 : 4) |
                        (((path.path.y & 0x1) == 0) ? 0 : 2) |
                        (((path.path.z & 0x1) == 0) ? 0 : 1) |
                        (((path.path.x & 0x2) == 0) ? 0 : 32) |
                        (((path.path.y & 0x2) == 0) ? 0 : 16) |
                        (((path.path.z & 0x2) == 0) ? 0 : 8));
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
                            nof_leaves += colors.get_leaves_count(level, dag.get_node(level, childIndex));
                        }
                    }
                }
                nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
            }
        }

        if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves))
        {
            return ColorType();
        }
        return colorLeaf.get_color(nof_leaves);
    }

    template<typename TDAG>
    EnclosedLeavesType fix_enclosed_leaves_impl(
            uint32 level,
            uint32 nodeIndex,
            TDAG& dag,
            StaticArray<EnclosedLeavesType>& enclosedLeaves,
            const uint32 topLevels)
    {
        if (level == dag.leaf_level())
        {
            const uint64 leaf = dag.get_leaf(nodeIndex).to_64();
            const uint32 count = Utils::popcll(leaf);
            check(count != 0 && count <= 64);
            return count;
        }
        else if (level >= topLevels)
        {
            const uint32 node = dag.get_node(level, nodeIndex);
            const uint32 count = node >> 8;
            if (count > 0)
            {
                return count;
            }
            else
            {
                const uint8 childMask = Utils::child_mask(node);
                uint64 newCount = 0;
                for (uint8 child = 0; child < 8; child++)
                {
                    if (childMask & (1 << child))
                    {
                        const uint32 childIndex = dag.get_child_index(level, nodeIndex, childMask, child);
                        const uint32 childCount = cast<uint32>(fix_enclosed_leaves_impl(level + 1, childIndex, dag, enclosedLeaves, topLevels));
                        check(childCount != 0);
                        newCount += childCount;
                    }
                }
                check(newCount != 0);
                checkInf(newCount, 1 << 24);
                if (count > 0)
                {
                    checkEqual(count, newCount);
                }

                uint32& nodeRef = dag.data[nodeIndex];
                check(node == nodeRef);
                nodeRef = uint32(newCount << 8) | childMask;

                check(childMask == Utils::child_mask(nodeRef));
                check(newCount == (nodeRef >> 8));
                return EnclosedLeavesType(newCount);
            }
        }
        else
        {
            const uint32 node = dag.get_node(level, nodeIndex);
            const uint8 childMask = Utils::child_mask(node);
            const uint32 mask = node >> 8;
            if (enclosedLeaves[mask] != 0)
            {
                // Already been there
                return enclosedLeaves[mask];
            }
            else
            {
                EnclosedLeavesType count = 0;
                for (uint8 child = 0; child < 8; child++)
                {
                    if (childMask & (1 << child))
                    {
                        const uint32 childIndex = dag.get_child_index(level, nodeIndex, childMask, child);
                        const EnclosedLeavesType childCount = fix_enclosed_leaves_impl(level + 1, childIndex, dag, enclosedLeaves, topLevels);
                        checkInfEqual(count, count + childCount); // Check for overflows
                        check(childCount != 0);
                        count += childCount;
                    }
                }
                enclosedLeaves[mask] = count;
                return count;
            }
        }
    }
    template<typename TDAG>
    void fix_enclosed_leaves(TDAG& dag, StaticArray<EnclosedLeavesType>& enclosedLeaves, uint32 topLevels)
    {
        SCOPED_STATS("fix_enclosed_leaves");

        printf("%" PRIu64 " enclosed leaves\n", enclosedLeaves.size());

        for (auto& enclosedLeave : enclosedLeaves)
            enclosedLeave = 0;

        fix_enclosed_leaves_impl(0, dag.get_first_node_index(), dag, enclosedLeaves, topLevels);
    }
}
