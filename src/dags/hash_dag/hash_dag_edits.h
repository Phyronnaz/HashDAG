#pragma once

#include "typedefs.h"
#include "color_utils.h"
#include "path.h"
#include "hash_dag_colors.h"
#include "hash_dag_utils.h"
#include "hash_table.h"
#include <memory>
#include <unordered_map>
#include <functional>

struct HashDagEdits
{
    static constexpr uint32 C_IgnoreColorIndex = 0xFFFFFFF;
    static constexpr uint32 C_RemoveColorIndex = 0xEFFFFFF;
    static constexpr uint32 C_InvalidNodeIndex = 0xDFFFFFF;

    struct EditIndex
    {
        uint32 nodeIndex;
        uint32 colorIndex;
    };
    struct LeafEditIndex
    {
        uint32 nodeIndex;
        uint32 nodeCount;
    };

    struct ColorToBuild
    {
        std::unique_ptr<ColorLeafBuilder> builder;
        CompressedColorLeaf oldLeaf;
        uint32 newLeafIndex = 0xFFFFFFFF;
    };

    struct EditStats
    {
        uint64 numNodes = 0;
        uint64 numVoxels = 0;

        inline void operator+=(const EditStats& other)
        {
            numNodes += other.numNodes;
            numVoxels += other.numVoxels;
        }
    };

    struct EditParameters
    {
        HashTable& hashTable;
        HashDAGColors& hashColors;
        std::vector<ColorToBuild> colorsToBuild = {};
        EditStats stats = {};
        LocalStatsRecorder statsRecorder;

        EditParameters(HashTable& hashTable, HashDAGColors& hashColors, StatsRecorder& statsRecorder)
                : hashTable(hashTable)
                , hashColors(hashColors)
                , statsRecorder(statsRecorder)
        {
        }
    };

    struct LeafEditParameters
    {
        HashTable& hashTable;

        const CompressedColorLeaf leaf; // might not be valid
        ColorLeafBuilder& leafBuilder;

        // Those are used to index the color array
        // before/after edit
        uint64 oldLeavesCount = 0;

        EditStats& stats;
        LocalStatsRecorder& statsRecorder;
    };

    struct Thread
    {
        explicit Thread(LocalStatsRecorder& statsRecorder)
                : statsRecorder(statsRecorder)
        {
        }

        std::unique_ptr<ColorLeafBuilder> builder;
        LocalStatsRecorder statsRecorder;
        EditStats stats; // per thread stats, merged at the end
        std::function<void()> lambda;
        uint32 finalIndex = 0;
    };
    struct ThreadedEditParameters : EditParameters
    {
        using EditParameters::EditParameters;

        std::unordered_map<uint3, std::shared_ptr<Thread>> threads = {};
    };

#if EDITS_COUNTERS
#define EDIT_COUNTER(...) __VA_ARGS__
#else
#define EDIT_COUNTER(...)
#endif

    template<bool Queue, typename T>
    static EditIndex edit_threaded(
            const uint32 level,
            const uint32 nodeIndex,
            uint32 colorNodeIndex,
            const Path path,
            const T& editor,
            ThreadedEditParameters& parameters)
    {
        check(level <= C_colorTreeLevels);

        auto& hashColors = parameters.hashColors;
        auto& hashTable = parameters.hashTable;
        constexpr uint32 levels = C_maxNumberOfLevels;
        EDIT_COUNTER(auto& stats = parameters.stats);

        EDIT_COUNTER(stats.numNodes++);

        if (!editor.should_edit(path, levels - level))
        {
            if (Queue)
            {
                return { C_InvalidNodeIndex, C_IgnoreColorIndex };
            }
            else
            {
                return { nodeIndex, C_IgnoreColorIndex };
            }
        }

        // Creates the color node when editing an empty space
        const auto allocateColorNode = [&]
        {
            check(colorNodeIndex == 0);
            colorNodeIndex = cast<uint32>(hashColors.allocate_interior_node());
            for (int index = 0; index < 7; index++)
            {
                hashColors.allocate_interior_node();
            }
        };

        const uint32* nodePtr;
        uint8 childMask;
        if (nodeIndex != C_InvalidNodeIndex)
        {
            nodePtr = hashTable.get_sys_ptr(level, nodeIndex);
            childMask = Utils::child_mask(nodePtr[0]);
        }
        else
        {
            nodePtr = nullptr;
            childMask = 0;
        }

        // Iterate children
        uint32 nodeChildOffset = 1;
        uint32 newChildren[8];
        bool childrenChanged = false;
        for (uint8 child = 0; child < 8; child++)
        {
            Path newPath = path;
            newPath.descend(child);
            uint32 childNodeIndex;
            if (childMask & (1u << child))
            {
                childNodeIndex = nodePtr[nodeChildOffset++];
            }
            else
            {
                childNodeIndex = C_InvalidNodeIndex;
            }

            uint32 newChildNodeIndex;

            uint32 childColorNodeIndex = 0;
            if (level == 0 || colorNodeIndex > 0) // Only level 0 color node has index 0
            {
                // Read color node if existing
                childColorNodeIndex = hashColors.get_node(colorNodeIndex + child);
            }

            if (level == C_colorTreeLevels - 1)
            {
                // Avoid creating empty color leaves for nothing
                if (editor.should_edit(newPath, levels - level - 1))
                {
                    CompressedColorLeaf existingColorLeaf;
                    uint32 existingColorLeafIndex = uint32(-1);
                    if (childColorNodeIndex > 0)
                    {
                        existingColorLeaf = hashColors.get_leaf(childColorNodeIndex);
                        existingColorLeafIndex = childColorNodeIndex;
                    }

                    if (Queue)
                    {
                        auto thread = std::make_shared<Thread>(parameters.statsRecorder);
                        thread->builder = std::make_unique<ColorLeafBuilder>();
                        thread->lambda = [threadPtr = std::weak_ptr<Thread>(thread), level, childNodeIndex, newPath, existingColorLeaf, &editor, hashTable = &parameters.hashTable]()
                        {
                        	PROFILE_SCOPE("Edit Leaf");
                            auto thread = threadPtr.lock();
                            LeafEditParameters childParameters{ *hashTable, existingColorLeaf, *thread->builder, 0, thread->stats, thread->statsRecorder };
                            thread->finalIndex = edit_leaf<C_colorTreeLevels>(childNodeIndex,
                                                                              newPath,
                                                                              editor,
                                                                              childParameters).nodeIndex;
                        };

                        check(parameters.threads.find(newPath.path) == parameters.threads.end());
                        parameters.threads.insert({ newPath.path, std::move(thread) });
                        newChildNodeIndex = C_InvalidNodeIndex;
                    }
                    else
                    {
                        const auto& thread = parameters.threads.at(newPath.path);
                        check(thread.get());
                        newChildNodeIndex = thread->finalIndex;

                        if (thread->builder->get_color_index() > 0)
                        {
                            if (existingColorLeafIndex == uint32(-1) || !Utils::has_flag(existingColorLeafIndex))
                            {
                                check(!existingColorLeaf.is_valid() || existingColorLeaf.is_shared());
                                check(childColorNodeIndex == 0 || existingColorLeaf.is_shared());

                                if (colorNodeIndex == 0)
                                {
                                    allocateColorNode();
                                }
                                check(colorNodeIndex != 0);

                                // Allocate leaf
                                childColorNodeIndex = hashColors.allocate_leaf();

                                // Update child ref
                                hashColors.set_node(colorNodeIndex + child, childColorNodeIndex);

                                existingColorLeafIndex = childColorNodeIndex;
                            }
                            check(existingColorLeafIndex != uint32(-1));

                            parameters.colorsToBuild.push_back(ColorToBuild{ std::move(thread->builder), existingColorLeaf, existingColorLeafIndex });
                        }
                    }
                }
                else
                {
                    newChildNodeIndex = Queue ? C_InvalidNodeIndex : childNodeIndex;
                }
            }
            else
            {
                check(level < C_colorTreeLevels - 1);

                const EditIndex childNodeIndices =
                        edit_threaded<Queue>(
                                level + 1,
                                childNodeIndex,
                                childColorNodeIndex,
                                newPath,
                                editor,
                                parameters);
                newChildNodeIndex = childNodeIndices.nodeIndex;

                if (Queue)
                {
                    check(childNodeIndices.nodeIndex == C_InvalidNodeIndex);
                    check(childNodeIndices.colorIndex == C_IgnoreColorIndex);
                }
                else
                {
                    if (childNodeIndices.colorIndex != childColorNodeIndex &&
                        childNodeIndices.colorIndex != C_IgnoreColorIndex &&
                        childNodeIndices.colorIndex != C_RemoveColorIndex)
                    {
                        check(childColorNodeIndex == 0);
                        if (level > 0 && colorNodeIndex == 0)
                        {
                            allocateColorNode();
                        }
                        hashColors.set_node(colorNodeIndex + child, childNodeIndices.colorIndex);
                    }
                }
            }

            if (Queue)
            {
                check(newChildNodeIndex == C_InvalidNodeIndex);
            }
            else
            {
                childrenChanged |= childNodeIndex != newChildNodeIndex;
                newChildren[child] = newChildNodeIndex;
            }
        }

        if (Queue)
        {
            return { C_InvalidNodeIndex, C_IgnoreColorIndex };
        }

        // If nothing changed, keep the same index
        if (!childrenChanged)
        {
            return { nodeIndex, colorNodeIndex };
        }

        // Else create a new node
        uint32 node[9];
        uint32 nodeSize = 0;
        node[nodeSize++] = 0; // no need for node counts above color leaves
        for (uint8 child = 0; child < 8; child++)
        {
            if (newChildren[child] != C_InvalidNodeIndex)
            {
                node[nodeSize++] = newChildren[child];
                node[0] |= 1u << child;
            }
        }

        // Empty
        if (Utils::child_mask(node[0]) == 0)
            return { C_InvalidNodeIndex, C_IgnoreColorIndex };

        const uint32 newIndex = hashTable.find_or_add_interior_node(level, nodeSize, node);
        return { newIndex, colorNodeIndex };
    }

    template<uint32 level, typename T>
    static LeafEditIndex edit_leaf(
            const uint32 nodeIndex,
            const Path path,
            const T& editor,
            LeafEditParameters& parameters)
    {
        PROFILE_FUNCTION_SLOW();
    	
        auto& hashTable = parameters.hashTable;
        constexpr uint32 levels = C_maxNumberOfLevels;
        constexpr uint32 leafLevel = C_leafLevel;
        EDIT_COUNTER(auto& stats = parameters.stats);
#if EDITS_ENABLE_COLORS
        auto& leaf = parameters.leaf;
        auto& leafBuilder = parameters.leafBuilder;
#endif

        EDIT_COUNTER(stats.numNodes++);

        // Check if the node is going to be full/empty, or if we don't want to edit it
        {
            EditScopeStat scopeStat(parameters.statsRecorder, EStatNames::EarlyExitChecks);
        	PROFILE_SCOPE_SLOW("EarlyExitChecks");

            const auto getVoxelCount = [&]()
            {
                if (level < leafLevel)
                {
                    const uint32* nodePtr = hashTable.get_sys_ptr(level, nodeIndex);
                    return nodePtr[0] >> 8;
                }
                else
                {
                    const uint32* nodePtr = hashTable.get_sys_ptr(level, nodeIndex);
                    return Utils::popc(nodePtr[0]) + Utils::popc(nodePtr[1]);
                }
            };

            const uint32 numVoxelsInFullNode = 1u << (3 * (levels - level)); // 8^depth

            if (!editor.should_edit(path, levels - level))
            {
                uint32 count;
                if (nodeIndex != C_InvalidNodeIndex)
                {
                    count = getVoxelCount();
#if EDITS_ENABLE_COLORS
                    check(leaf.is_valid());
                    uint64 start = parameters.oldLeavesCount;
                    if (leaf.is_shared())
                    {
                        start += leaf.get_offset();
                    }

                    scopeStat.pause();
                    {
                        EditScopeStat colorScope(parameters.statsRecorder, EStatNames::SkipEdit_CopyColors);
                        leaf.copy_colors(leafBuilder, start, count);
                    }
                    scopeStat.resume();

                    parameters.oldLeavesCount += count;
#endif
                }
                else
                {
                    count = 0;
                }

                return { nodeIndex, count };
            }


            if (editor.is_empty(path, levels - level))
            {
#if EDITS_ENABLE_COLORS
                if (nodeIndex != C_InvalidNodeIndex)
                {
                    parameters.oldLeavesCount += getVoxelCount();
                }
#endif
                EDIT_COUNTER(stats.numVoxels += numVoxelsInFullNode);
                return { C_InvalidNodeIndex, 0 };
            }

#if ADD_FULL_NODES_FIRST && EARLY_FULL_CHECK
            if (editor.is_full(path, levels - level))
            {
                if (level < leafLevel)
                {
                    checkEqual(hashTable.get_sys_ptr(level, parameters.hashTable.get_full_node_index(level))[0] >> 8, numVoxelsInFullNode);
                }
#if EDITS_ENABLE_COLORS
                if (nodeIndex != C_InvalidNodeIndex)
                {
                    parameters.oldLeavesCount += getVoxelCount();
                }
                scopeStat.pause();
                {
                    EditScopeStat colorScope(parameters.statsRecorder, EStatNames::EntirelyFull_AddColors);
                    leafBuilder.add_large_single_color(editor.get_single_color(), numVoxelsInFullNode);
                }
                scopeStat.resume();
#endif
                EDIT_COUNTER(stats.numVoxels += numVoxelsInFullNode);
                return { parameters.hashTable.get_full_node_index(level), numVoxelsInFullNode };
            }
#endif
        }

#if defined(__CUDACC__) && !defined(__PARSER__) // no C++17 support
        if (true)
#else
        if constexpr (level == C_leafLevel)
#endif
        {
            EditScopeStat scopeStat(parameters.statsRecorder, EStatNames::LeafEdit);

            uint64 leafMask;
            if (nodeIndex != C_InvalidNodeIndex)
            {
                const uint32* nodePtr = hashTable.get_sys_ptr(level, nodeIndex);
                std::memcpy(&leafMask, nodePtr, sizeof(uint64));
            }
            else
            {
                leafMask = 0;
            }
            const uint64 initialLeafMask = leafMask;

            // Iterate level 1 children
            for (uint8 child1 = 0; child1 < 8; child1++)
            {
                Path newPath1 = path;
                newPath1.descend(child1);
                // Iterate level 2 children
                for (uint8 child2 = 0; child2 < 8; child2++)
                {
                    Path newPath2 = newPath1;
                    newPath2.descend(child2);
                    const float3 position = newPath2.as_position(0);

                    const uint32 bitIndex = child1 * 8u + child2;
                    check(bitIndex < 64);
                    const uint64 mask = uint64(1) << bitIndex;

                    const bool shouldEdit = editor.should_edit_impl(position, position + 1);
                    const bool previousValue = leafMask & mask;
                    const bool newValue =
                            shouldEdit
                            ? editor.get_new_value(position, previousValue)
                            : previousValue;

                    EDIT_COUNTER(stats.numVoxels += shouldEdit);

#if EDITS_ENABLE_COLORS
                    if (newValue)
                    {
                        CompressedColor color;
                        auto get_previous_color = [&]()
                        {
                            if (previousValue)
							{
								if (ensure(leaf.is_valid()) &&
									ensure(leaf.is_valid_index(parameters.oldLeavesCount))) /* not really needed, but useful when dev with invalid stuff */
								{
									return leaf.get_color(parameters.oldLeavesCount);
								}
							}
							return CompressedColor{};
						};
                        if (shouldEdit)
                        {
                            color = editor.get_new_color(position, get_previous_color, previousValue, newValue);
                        }
                        else
                        {
                            color = get_previous_color();
                        }
                        leafBuilder.add(color);
                    }
                    parameters.oldLeavesCount += previousValue;
#endif

                    if (previousValue != newValue)
                    {
                        if (newValue)
                        {
                            leafMask |= mask;
                        }
                        else
                        {
                            leafMask &= ~mask;
                        }
                    }
                }
            }

            if (leafMask == initialLeafMask)
                return { nodeIndex, Utils::popcll(leafMask) };

            // Empty
            if (leafMask == 0)
                return { C_InvalidNodeIndex, 0 };

            scopeStat.pause();

            EditScopeStat findStat(parameters.statsRecorder, EStatNames::FindOrAddLeaf);
            const uint32 index = hashTable.find_or_add_leaf_node(level, leafMask);
            return { index , Utils::popcll(leafMask) };
        }
        else
        {
            EditScopeStat scopeStat(parameters.statsRecorder, EStatNames::InteriorEdit);

            const uint32* nodePtr;
            uint8 childMask;
            if (nodeIndex != C_InvalidNodeIndex)
            {
                nodePtr = hashTable.get_sys_ptr(level, nodeIndex);
                childMask = Utils::child_mask(nodePtr[0]);
            }
            else
            {
                nodePtr = nullptr;
                childMask = 0;
            }

            // Iterate children
            uint32 nodeChildOffset = 1;
            uint32 newChildren[8];
            uint32 newNodeCount = 0;
            bool childrenChanged = false;
            for (uint8 child = 0; child < 8; child++)
            {
                Path newPath = path;
                newPath.descend(child);
                uint32 childNodeIndex;
                if (childMask & (1u << child))
                {
                    childNodeIndex = nodePtr[nodeChildOffset++];
                }
                else
                {
                    childNodeIndex = C_InvalidNodeIndex;
                }

                scopeStat.pause();
                const LeafEditIndex newChildNodeIndexAndCount = edit_leaf<level + 1>(
                        childNodeIndex,
                        newPath,
                        editor,
                        parameters);
                scopeStat.resume();

                const uint32 newChildNodeIndex = newChildNodeIndexAndCount.nodeIndex;
                if (newChildNodeIndex != C_InvalidNodeIndex)
                {
                    check(newChildNodeIndexAndCount.nodeCount > 0);
                    newNodeCount += newChildNodeIndexAndCount.nodeCount;
                }
                else
                {
                    check(newChildNodeIndexAndCount.nodeCount == 0);
                }

                childrenChanged |= childNodeIndex != newChildNodeIndex;
                newChildren[child] = newChildNodeIndex;
            }
            checkf(newNodeCount < (1u << 24), "%u subnodes, can't store it in the node. Need to decrease the size of the color leaves.", newNodeCount);

            // If nothing changed, keep the same index
            if (!childrenChanged)
            {
                if (nodeIndex != C_InvalidNodeIndex)
                {
                    checkEqual(nodePtr[0] >> 8u, newNodeCount);
                    checkEqual(newNodeCount, HashDagUtils::count_children(hashTable, level, levels, nodeIndex));
                }
                return { nodeIndex, newNodeCount };
            }

            // Else create a new node
            uint32 node[9];
            uint32 nodeSize = 0;
            node[nodeSize++] = (newNodeCount << 8u);
            for (uint8 child = 0; child < 8; child++)
            {
                if (newChildren[child] != C_InvalidNodeIndex)
                {
                    node[nodeSize++] = newChildren[child];
                    node[0] |= 1u << child;
                }
            }

            // Empty
            if (Utils::child_mask(node[0]) == 0)
                return { C_InvalidNodeIndex, 0 };

            scopeStat.pause();

            EditScopeStat findAddStat(parameters.statsRecorder, EStatNames::FindOrAddInterior);

            const uint32 newIndex = hashTable.find_or_add_interior_node(level, nodeSize, node);
            checkEqual(newNodeCount, HashDagUtils::count_children(hashTable, level, levels, newIndex));
            return { newIndex, newNodeCount };
        }
    }
};
