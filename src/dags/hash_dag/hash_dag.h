#pragma once

#include "typedefs.h"

#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>

#include "path.h"
#include "stats.h"
#include "dags/base_dag.h"
#include "dags/hash_dag/hash_dag_globals.h"
#include "dags/hash_dag/hash_table.h"
#include "dags/hash_dag/hash_dag_utils.h"
#include "dags/hash_dag/hash_dag_edits.h"
#include "dags/hash_dag/hash_dag_colors.h"

class HashDAGUndoRedo
{
public:
	struct ColorLeafDiff
	{
		uint32 leafIndex = 0;
		CompressedColorLeaf previousLeaf;
		CompressedColorLeaf newLeaf;
	};
	struct Frame
	{
		void delete_frame()
		{
			for (auto& leaf : colorLeavesDiff)
			{
				if (!leaf.previousLeaf.is_shared())
				{
					leaf.previousLeaf.free();
				}
			}
			colorLeavesDiff.resize(0);
		}

		uint32 firstNodeIndex = 0;
		std::vector<ColorLeafDiff> colorLeavesDiff;
	};

	std::vector<Frame> undoFrames;
	std::vector<Frame> redoFrames;

    double get_total_used_memory() const
    {
        return get_total_used_memory(undoFrames) + get_total_used_memory(redoFrames);
    }

	void add_frame(const Frame& frame)
	{
        PROFILE_FUNCTION();
    	
		for (auto& redoFrame : redoFrames)
		{
			redoFrame.delete_frame();
		}
		redoFrames.resize(0);

		undoFrames.push_back(frame);
	}
	template<typename TDAG, typename TColors>
	void undo(TDAG& dag, TColors& colors)
	{
		if (undoFrames.empty())
			return;

		const Frame undoFrame = undoFrames.back();
		undoFrames.pop_back();

		Frame redoFrame;
		redoFrame.firstNodeIndex = dag.firstNodeIndex;
		redoFrame.colorLeavesDiff = invert_diff(undoFrame.colorLeavesDiff);

		dag.firstNodeIndex = undoFrame.firstNodeIndex;
		for (auto& leaf : undoFrame.colorLeavesDiff)
		{
			auto& ref = colors.get_leaf_ref(leaf.leafIndex);
			check(ref == leaf.newLeaf);
			ref = leaf.previousLeaf;

			colors.upload_leaf_index_to_gpu_async(leaf.leafIndex);
		}

		redoFrames.push_back(redoFrame);

		cudaDeviceSynchronize();
	}
	template<typename TDAG, typename TColors>
	void redo(TDAG& dag, TColors& colors)
	{
		if (redoFrames.empty())
			return;

		const Frame redoFrame = redoFrames.back();
		redoFrames.pop_back();

		Frame undoFrame;
		undoFrame.firstNodeIndex = dag.firstNodeIndex;
		undoFrame.colorLeavesDiff = invert_diff(redoFrame.colorLeavesDiff);

		dag.firstNodeIndex = redoFrame.firstNodeIndex;
		for (auto& leaf : redoFrame.colorLeavesDiff)
		{
			auto& ref = colors.get_leaf_ref(leaf.leafIndex);
			check(ref == leaf.newLeaf);
			ref = leaf.previousLeaf;

			colors.upload_leaf_index_to_gpu_async(leaf.leafIndex);
		}

		undoFrames.push_back(undoFrame);

		cudaDeviceSynchronize();
	}

	void print_stats()
	{
		printf("##############################################\n");
		printf("Undo Redo stats:\n");
		printf("Undo stack:\n");
		print_stats(undoFrames);
		printf("Redo stack:\n");
		print_stats(redoFrames);
		printf("##############################################\n");
	}

	void free()
	{
		for (auto& frame : undoFrames)
		{
			frame.delete_frame();
		}
		for (auto& frame : redoFrames)
		{
			frame.delete_frame();
		}
		undoFrames.resize(0);
		redoFrames.resize(0);
	}

private:
	static std::vector<ColorLeafDiff> invert_diff(const std::vector<ColorLeafDiff>& diff)
	{
		std::vector<ColorLeafDiff> result;
		result.reserve(diff.size());
		for (auto& leaf : diff)
		{
			ColorLeafDiff newDiff;
			newDiff.leafIndex = leaf.leafIndex;
			newDiff.previousLeaf = leaf.newLeaf;
			newDiff.newLeaf = leaf.previousLeaf;
			result.push_back(newDiff);
		}
		return result;
	}
    static double get_total_used_memory(const std::vector<Frame>& frames)
    {
        double usage = 0;
		for (auto& frame : frames)
		{
			for (auto& diff : frame.colorLeavesDiff)
			{
				if (!diff.previousLeaf.is_shared())
				{
					usage += diff.previousLeaf.blocks_CPU.size_in_MB();
					usage += diff.previousLeaf.weights_CPU.size_in_MB();
					usage += diff.previousLeaf.macroBlocks_CPU.size_in_MB();
				}
			}
		}
        return usage;
    }
	static void print_stats(const std::vector<Frame>& frames)
	{
		printf("\t%u frames\n", (uint32)frames.size());

		double blocksSizeInMB = 0;
		double weightsSizeInMB = 0;
		double macroBlocksSizeInMB = 0;
		uint32 numLeaves = 0;
		for (auto& frame : frames)
		{
			for (auto& diff : frame.colorLeavesDiff)
			{
				if (!diff.previousLeaf.is_shared())
				{
					numLeaves++;
					blocksSizeInMB += diff.previousLeaf.blocks_CPU.size_in_MB();
					weightsSizeInMB += diff.previousLeaf.weights_CPU.size_in_MB();
					macroBlocksSizeInMB += diff.previousLeaf.macroBlocks_CPU.size_in_MB();
				}
			}
		}
		printf(
			"\t%u owned leaves\n"
			"\t%fMB used by blocks\n"
			"\t%fMB used by weights\n"
			"\t%fMB used by macro blocks\n"
			"\tTotal: %fMB\n",
			numLeaves,
			blocksSizeInMB,
			weightsSizeInMB,
			macroBlocksSizeInMB,
			blocksSizeInMB + weightsSizeInMB + macroBlocksSizeInMB);
	}
};

struct HashDAG : BaseDAG
{
#if !MANUAL_VIRTUAL_MEMORY
    // hack for faster access, is data.pool
    uint32* __restrict__ pool;
#endif
    HashTable data;
    uint32 firstNodeIndex = 0;

    HOST_DEVICE bool is_valid() const
    {
        return data.is_valid();
    }

    HOST_DEVICE uint32 get_first_node_index() const
    {
        return firstNodeIndex;
    }

#define get_ptr(level, ptr) get_ptr_impl(level, ptr, __LINE__, __FILE__)
    HOST_DEVICE uint32 get_node(uint32 level, uint32 index) const
    {
        checkInf(level, leaf_level());
        return *get_ptr(level, index);
    }
    HOST_DEVICE uint32 get_child_index(uint32 level, uint32 index, uint8 childMask, uint8 child) const
    {
        return *get_ptr(level, index + Utils::child_offset(childMask, child));
    }
    HOST_DEVICE Leaf get_leaf(uint32 index) const
    {
        const uint32* node = get_ptr(leaf_level(), index);
        return { node[0], node[1] };
    }
#undef get_ptr
	HOST void free()
    {
	    data.destroy();
    }

public:
    template<typename T>
    HOST void edit_threads(const T& editor, HashDAGColors& hashColors, HashDAGUndoRedo& undoRedo, StatsRecorder& statsRecorder)
    {
        PROFILE_FUNCTION();
    	
        BasicStats stats;

        SimpleScopeStat editsStats;

#if UNDO_REDO
        HashDAGUndoRedo::Frame frame;
        frame.firstNodeIndex = firstNodeIndex;
#endif

        HashDagEdits::ThreadedEditParameters parameters{ data, hashColors, statsRecorder };
        // Make sure all stats appear in the frame
        parameters.statsRecorder.report(EStatNames::EarlyExitChecks, 0);
        parameters.statsRecorder.report(EStatNames::EntirelyFull_AddColors, 0);
        parameters.statsRecorder.report(EStatNames::SkipEdit_CopyColors, 0);
        parameters.statsRecorder.report(EStatNames::LeafEdit, 0);
        parameters.statsRecorder.report(EStatNames::FindOrAddLeaf, 0);
        parameters.statsRecorder.report(EStatNames::InteriorEdit, 0);
        parameters.statsRecorder.report(EStatNames::FindOrAddInterior, 0);

        // First create the threads functions in a first pass
        {
        	PROFILE_SCOPE("First Pass");
	        stats.start_work("first pass");
	        HashDagEdits::edit_threaded<true>(0, firstNodeIndex, 0, Path(0, 0, 0), editor, parameters);
	        stats.flush(statsRecorder);
        }
        	
        // NOTE: we don't start the threads in edit_threaded to be able to measure the time of edit_threaded without the noise added by starting threads

#if THREADED_EDITS
        std::vector<std::thread> threads;

		std::vector<std::function<void()>*> tasks;
		std::atomic<uint32> taskCounter{ 0 };
    	
		// Start the threads
        {
        	PROFILE_SCOPE("Start Threads");
	        stats.start_work("start threads");
	#if NUM_THREADS == 0
	        threads.reserve(parameters.threads.size());
	        for (auto& it : parameters.threads)
	        {
	            threads.emplace_back(it.second->lambda);
	        }
	#else
	        tasks.reserve(parameters.threads.size());
	        for (auto& it : parameters.threads)
	        {
	            tasks.emplace_back(&it.second->lambda);
	        }

	        threads.reserve(NUM_THREADS);
	        const auto lambda = [&]()
	        {
	        	NAME_THREAD("Edit Thread");
	        	PROFILE_SCOPE("Edit Thread");
	            for(;;)
	            {
	                const uint32 taskIndex = taskCounter.fetch_add(1);
	                if (taskIndex >= parameters.threads.size()) return;
	                (*tasks[taskIndex])();
	                if (taskIndex % 100 == 0 && !BENCHMARK)
	                {
	                    EDIT_TIMES(printf("Task %u/%u    \r", taskIndex, uint32(parameters.threads.size())));
	                }
	            }
	        };
	        for (uint32 index = 0; index < NUM_THREADS; index++)
	        {
	            threads.emplace_back(lambda);
	        }
	#endif
	        stats.flush(statsRecorder);
        }

        // Then wait for them
        {
        	PROFILE_SCOPE("Wait");
	        stats.start_work("waiting");
	        for (auto& thread : threads)
	        {
	            thread.join();
	        }
	        stats.flush(statsRecorder);
		}
#else
		stats.start_work("running tasks");
		for (auto& it : parameters.threads)
		{
			it.second->lambda();
		}
		stats.flush(statsRecorder);
#endif

		for (auto& it : parameters.threads)
		{
			parameters.stats += it.second->stats;
		}

        // Finally in a second pass apply their results
        {
        	PROFILE_SCOPE("Second Pass");
	        stats.start_work("second pass");
	        firstNodeIndex = HashDagEdits::edit_threaded<false>(0, firstNodeIndex, 0, Path(0, 0, 0), editor, parameters).nodeIndex;
	        stats.flush(statsRecorder);
        }

        statsRecorder.report("edits", editsStats.get_time());

        EDIT_TIMES(printf("Done! %fms                     \n", editsStats.get_time()));

        statsRecorder.report("num threads", parameters.threads.size());
        statsRecorder.report("num nodes", parameters.stats.numNodes);
		auto const count = editor.edit_count();
		if( ~uint32(0) == count )
			statsRecorder.report("num voxels", parameters.stats.numVoxels);
		else
			statsRecorder.report("num voxels", count);
        statsRecorder.report("num color leaves rebuilt", parameters.colorsToBuild.size());

        {
        	PROFILE_SCOPE("Rebuilding Colors");
	        stats.start_work("rebuilding colors");
	        for (auto& colorToBuild : parameters.colorsToBuild)
	        {
	#if UNDO_REDO
	            HashDAGUndoRedo::ColorLeafDiff diff;
	            diff.leafIndex = colorToBuild.newLeafIndex;
	            diff.previousLeaf = colorToBuild.oldLeaf;
	#endif
	            auto& leaf = hashColors.get_leaf_ref(colorToBuild.newLeafIndex);
	            colorToBuild.builder->build(leaf, !UNDO_REDO);
	#if UNDO_REDO
	            diff.newLeaf = leaf;
	            frame.colorLeavesDiff.push_back(diff);
	#endif
	        }
	        hashColors.upload_to_gpu(false);
	        stats.flush(statsRecorder);
        }

#if UNDO_REDO
        undoRedo.add_frame(frame);
#endif
    }

    // Max level: included
    HOST void remove_stale_nodes(uint32 maxLevel)
    {
        PROFILE_FUNCTION();
    	
        const double memoryUsageBeforeGC = data.get_virtual_used_size(false);
        {
            SCOPED_STATS("GC");

            check(maxLevel < C_maxNumberOfLevels);

            Stats stats;

            stats.start_work("check_nodes");
//            check_nodes();

            // First find valid nodes
            std::vector<std::unordered_set<uint32>> validNodes(maxLevel + 1);
            validNodes[0].emplace(firstNodeIndex);
            for (uint32 level = 0; level < maxLevel; level++)
            {
                stats.start_level_work(level + 1, "find valid nodes");
                validNodes[level + 1].reserve(data.get_level_size(level + 1));
                for (uint32 nodeIndex : validNodes[level])
                {
                    const uint32* node = data.get_sys_ptr(level, nodeIndex);
                    for (uint32 offset = 1; offset < Utils::total_size(node[0]); offset++)
                    {
                        validNodes[level + 1].emplace(node[offset]);
                    }
                }
            }

            // Then rebuild the levels, bottom up to have valid hashes, only keeping the valid nodes
            std::unordered_map<uint32, uint32> map;
            std::unordered_map<uint32, uint32> newMap;
            for (uint32 level = maxLevel; level != uint32(-1); level--)
            {
                stats.start_level_work(level, "rebuild - copying valid nodes");
                const bool isLeaf = level == leaf_level();
                /**
                 * newMap: old index -> tempNodes index or final index
                 * mapKeysToFix: list of old indices that are still pointing to tempNodes
                 */
                std::vector<uint32> tempNodes;
                std::vector<uint32> mapKeysToFix;

                // First fill tempNodes and mapKeysToFix
                newMap.reserve(validNodes[level].size());
                for (uint32 nodeIndex : validNodes[level])
                {
                    uint32* node = data.get_sys_ptr(level, nodeIndex);
                    const uint32 nodeSize = isLeaf ? 2 : Utils::total_size(node[0]);

                    // Update children indices
                    if (level != maxLevel)
                    {
                        check(!isLeaf);
                        for (uint32 offset = 1; offset < nodeSize; offset++)
                        {
                            node[offset] = map.at(node[offset]);
                        }
                    }

                    // Copy the node to tempNodes
                    const uint32 tempNodesTop = uint32(tempNodes.size());
                    tempNodes.resize(tempNodesTop + nodeSize);
                    for (uint32 offset = 0; offset < nodeSize; offset++)
                    {
                        tempNodes[tempNodesTop + offset] = node[offset];
                    }

                    check(node[0] == tempNodes[tempNodesTop]);
                    check(node[nodeSize - 1] == tempNodes[tempNodesTop + nodeSize - 1]);

                    // Set the newMap index to the tempNodes index
                    check(newMap.find(nodeIndex) == newMap.end());
                    newMap[nodeIndex] = tempNodesTop;
                    mapKeysToFix.push_back(nodeIndex);
                }
                // Empty all buckets
                for (uint32 bucket = 0; bucket < HashDagUtils::get_buckets_per_level(level); bucket++)
                {
#if USE_BLOOM_FILTER
                    auto const baseVirtualPtr = HashDagUtils::make_ptr(level, bucket, 0);
                    for (uint32 pindex = 0; pindex < data.get_bucket_size(level, bucket); pindex += C_pageSize)
                    {
                        auto const pageVirtualPtr = baseVirtualPtr + pindex;
                        auto const page = HashDagUtils::get_page(pageVirtualPtr);
                        data.bloom_filter_reset(page);
                    }
#endif
                    *data.get_bucket_size_ptr(level, bucket) = 0;
                }

#if ADD_FULL_NODES_FIRST
                // First add full nodes
                data.add_full_node(level);
#endif

                stats.start_level_work(level, "rebuild - adding nodes back");
                // Then add back the data, and assign the right indices in the map
                for (uint32 oldIndex : mapKeysToFix)
                {
                    uint32& mapValue = newMap.at(oldIndex); // mapValue: index into temp nodes, and now index to new position
                    if (isLeaf)
                    {
                        const uint64 leaf = Leaf{ tempNodes[mapValue + 0], tempNodes[mapValue + 1] }.to_64();
                        const uint32 hash = HashDagUtils::hash_leaf(leaf);
#if USE_BLOOM_FILTER
                        BloomFilter filter;
                        HashTable::bloom_filter_init_leaf(filter, leaf);
#endif
                        mapValue = data.add_leaf_node(level, leaf, hash BLOOM_FILTER_ARG(filter));
                    }
                    else
                    {
                        const uint32* nodePtr = &tempNodes[mapValue];
                        const uint32 nodeSize = Utils::total_size(nodePtr[0]);
                        const uint32 hash = HashDagUtils::hash_interior(nodeSize, nodePtr);
#if USE_BLOOM_FILTER
                        BloomFilter filter;
                        HashTable::bloom_filter_init_interior(filter, nodeSize, nodePtr);
#endif
                        mapValue = data.add_interior_node(level, nodeSize, nodePtr, hash BLOOM_FILTER_ARG(filter));
                    }
                }

                // Swap maps
                map = std::move(newMap);
                check(newMap.empty());
            }

            stats.start_work("check_nodes");
            check_nodes();
        }
        const double memoryUsageAfterGC = data.get_virtual_used_size(false);
        printf("Virtual memory used Before: %fMB; After: %fMB; Saved: %fMB\n", memoryUsageBeforeGC, memoryUsageAfterGC, memoryUsageBeforeGC - memoryUsageAfterGC);

        data.full_upload_to_gpu();
    }

    HOST void simulate_remove_stale_nodes(StatsRecorder& statsRecorder)
    {
        PROFILE_FUNCTION();
        SCOPED_STATS("Simulating GC");
        Stats stats;

        double memoryUsageWithGC = 0;
        double memoryUsageWithoutGC = 0;
        std::vector<std::unordered_set<uint32>> validNodes(C_leafLevel + 1);
        validNodes[0].emplace(firstNodeIndex);

        for (uint32 level = 0; level < C_leafLevel; level++)
        {
            stats.start_level_work(level + 1, "find valid nodes");
            validNodes[level + 1].reserve(data.get_level_size(level + 1));
#if ADD_FULL_NODES_FIRST
            validNodes[level + 1].emplace(data.get_full_node_index(level + 1));
#endif
            for (uint32 nodeIndex : validNodes[level])
            {
                const uint32* node = data.get_sys_ptr(level, nodeIndex);
                memoryUsageWithGC += Utils::to_MB(Utils::total_size(node[0]) * sizeof(uint32));
                for (uint32 offset = 1; offset < Utils::total_size(node[0]); offset++)
                {
                    validNodes[level + 1].emplace(node[offset]);
                }
            }
            memoryUsageWithoutGC += Utils::to_MB(data.get_level_size(level) * sizeof(uint32));
            statsRecorder.report("GC freed memory level " + std::to_string(level), memoryUsageWithoutGC - memoryUsageWithGC);
            printf("Freed memory: %f\n", memoryUsageWithoutGC - memoryUsageWithGC);
        }

        memoryUsageWithoutGC += Utils::to_MB(data.get_level_size(C_leafLevel) * sizeof(uint32));
        memoryUsageWithGC += Utils::to_MB(validNodes[C_leafLevel].size() * 2 * sizeof(uint32));

        statsRecorder.report("GC freed memory level " + std::to_string(C_leafLevel), memoryUsageWithoutGC - memoryUsageWithGC);
        statsRecorder.report("GC freed memory leaf level", memoryUsageWithoutGC - memoryUsageWithGC);
        printf("Total freed memory: %f\n", memoryUsageWithoutGC - memoryUsageWithGC);
    }

    HOST void check_nodes()
    {
        PROFILE_FUNCTION();
#if ENABLE_CHECKS
        std::unordered_set<uint32> queuedIndices;
        std::unordered_set<uint32> newQueuedIndices;
        queuedIndices.emplace(firstNodeIndex);

        printf("Checking level ");
        for (uint32 level = 0; level < levels - 1; level++)
        {
            printf("%u/%u... ", level, levels - 2);
            for (uint32 index : queuedIndices)
            {
                const uint32* ptr = data.get_sys_ptr(level, index);
                if (level != leaf_level())
                {
                    for (uint32 offset = 1; offset < Utils::total_size(ptr[0]); offset++)
                    {
                        newQueuedIndices.emplace(ptr[offset]);
                    }
                }
                else
                {
                    check(ptr[0] != 0 || ptr[1] != 0);
                }
                if (C_colorTreeLevels <= level && level < leaf_level())
                {
                    checkEqual(ptr[0] >> 8u, HashDagUtils::count_children(data, level, levels, index));
                }
            }
            queuedIndices = std::move(newQueuedIndices);
            check(newQueuedIndices.empty());
        }
        printf("Success!\n");
#endif
    }

private:
    HOST_DEVICE const uint32* get_ptr_impl(uint32 level, uint32 index, uint32 debugLine, const char* debugFile) const
    {
#if !MANUAL_VIRTUAL_MEMORY && defined(__CUDA_ARCH__)
        // hack for faster access
        return pool + index;
#endif
        return data.get_sys_ptr_impl(level, index, debugLine, debugFile);
    }
};
