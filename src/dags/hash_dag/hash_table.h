#pragma once

#include "typedefs.h"
#include "hash_dag_globals.h"
#include "stats.h"

#include <mutex>
#include <random>
#include <cstring>
#include <array>

#if BLOOM_FILTER_AVX || ALTERNATE_NODEFN_AVX_LEAFSEARCH
#include <x86intrin.h> // For *nix, Windows uses different header (<intrin.h>?)
#endif

namespace HashDagUtils
{
    // returns the index of bucket in an array of CTotalNumberOfBuckets
    HOST_DEVICE uint32 get_bucket_global_index(uint32 level, uint32 bucket)
    {
		check(level < C_maxNumberOfLevels);
        uint32 index;
        if (level < C_bucketsNumTopLevels)
        {
		    check(bucket < C_bucketsPerTopLevel);
            index = level * C_bucketsPerTopLevel;
        }
        else
        {
		    check(bucket < C_bucketsPerLowLevel);
            index = C_bucketsNumTopLevels * C_bucketsPerTopLevel + (level - C_bucketsNumTopLevels) * C_bucketsPerLowLevel;
        }
        index += bucket;
        check(index < C_totalNumberOfBuckets);
        return index;
    }
    HOST_DEVICE uint32 get_buckets_per_level(uint32 level)
    {
        return level < C_bucketsNumTopLevels ? C_bucketsPerTopLevel : C_bucketsPerLowLevel;
    }
    HOST_DEVICE uint32 get_bucket_size(uint32 level)
    {
        return level < C_bucketsNumTopLevels ? C_bucketSizeForTopLevels : C_bucketSizeForLowLevels;
    }
    HOST_DEVICE uint32 make_ptr(uint32 level, uint32 bucket, uint32 bucketPosition)
    {
		check(level < C_maxNumberOfLevels);
        uint32 index;
        if (level < C_bucketsNumTopLevels)
        {
		    check(bucket < C_bucketsPerTopLevel);
            index = (level * C_bucketsPerTopLevel + bucket) * C_bucketSizeForTopLevels;
        }
        else
        {
		    check(bucket < C_bucketsPerLowLevel);
            index = C_bucketsNumTopLevels * C_bucketsPerTopLevel * C_bucketSizeForTopLevels;
            index += ((level - C_bucketsNumTopLevels) * C_bucketsPerLowLevel + bucket) * C_bucketSizeForLowLevels;
        }
        index += bucketPosition;
        checkInf(index, C_totalVirtualAddresses);
        return index;
    }
    HOST_DEVICE uint32 get_page(uint32 ptr)
    {
        return ptr / C_pageSize;
    }
	HOST_DEVICE uint32 get_page_offset(uint32 ptr)
    {
	    return ptr % C_pageSize;
    }
    HOST_DEVICE uint32 get_bucket_from_hash(uint32 level, uint32 hash)
    {
        uint32 const bucketsPerLevel = get_buckets_per_level(level);
        checkf((bucketsPerLevel & (bucketsPerLevel - 1)) == 0, "bucketsPerLevel: %u", bucketsPerLevel);
        return hash & (bucketsPerLevel - 1);
    }


	HOST_DEVICE uint32 hash_leaf(uint64 leaf)
    {
    	PROFILE_FUNCTION_SLOW();
        return uint32(Utils::murmurhash64(leaf));
    }
    HOST_DEVICE uint32 hash_interior(const uint32 nodeSize, const uint32*__restrict__ node)
    {
    	PROFILE_FUNCTION_SLOW();
        return Utils::murmurhash32xN(node, nodeSize);
    }

#if USE_BLOOM_FILTER
    /* For the Bloom filter, we need to use hashes *different* from the
     * table entry for which the filter applies. Otherwise, we have a much
     * higher probability that some bits are set.
     *
     * For non-leaf pointers, we use the fasthash64(), slightly adapted
     * to handle our 32-bit word input. For the 64-bit fixed size leafs,
     * we use a 64-bit bit mixing function (hash64shift).
     */
    HOST_DEVICE uint64 bloom_hash_leaf(const uint64 leaf)
    {
        return Utils::hash64shift(leaf);
    }
    HOST_DEVICE uint64 bloom_hash_interior(const uint32 nodeSize, const uint32* __restrict__ node)
    {
        return Utils::fasthash64(node, nodeSize);
    }
#endif
}

#if !defined(__CUDA_ARCH__) && MANUAL_CPU_DATA
#define pageTable cpuData.cpuPageTable
#define pool cpuData.cpuPool
#define HAS_PAGE_TABLE 1
#else
#if MANUAL_VIRTUAL_MEMORY
#define pageTable gpuPageTable
#define HAS_PAGE_TABLE 1
#else
#define HAS_PAGE_TABLE 0
#endif
#define pool gpuPool
#endif

#if USE_BLOOM_FILTER
#undef USE_BLOOM_FILTER
#define USE_BLOOM_FILTER HAS_PAGE_TABLE
#endif

#if USE_BLOOM_FILTER
#define BLOOM_FILTER_ARG(x) , x
using BloomFilter = std::array<uint32, BLOOM_FILTER_WORDS>;
#else  // ! USE_BLOOM_FILTER
#define BLOOM_FILTER_ARG(x)
#endif // ~ USE_BLOOM_FILTER

struct HashTable
{
	HashTable() = default;

    HOST_DEVICE bool is_valid() const { return pool != nullptr; }

public:
	void create(uint32 poolSizeInPages);
	void destroy();
	void prefetch();
	void upload_to_gpu();
	void full_upload_to_gpu();
	void save_bucket_sizes(bool beforeEdits) const;
	// Size used by the nodes
	double get_virtual_used_size(bool printPerLevelUsage) const;
	uint32 get_level_size(uint32 level) const;

public:
#define get_sys_ptr(level, ptr) get_sys_ptr_impl(level, ptr, __LINE__, __FILE__)
	HOST_DEVICE uint32* get_sys_ptr_impl(uint32 level, uint32 ptr, uint32 debugLine, const char* debugFile) const
	{
#if HAS_PAGE_TABLE
        const uint32 page = HashDagUtils::get_page(ptr);
        const uint32 offset = HashDagUtils::get_page_offset(ptr);
        check(pageTable && pool);
		checkInf(page, pageTableSize);
        checkf(page < pageTableSize, "Address: %u; Level: %u; %s:%u", ptr, level, debugFile, debugLine);
		checkInf(offset, C_pageSize);
		checkf(pageTable[page] != 0, "Address: %u; Level: %u; %s:%u", ptr, level, debugFile, debugLine);
		uint32* result = pool + pageTable[page] * uint64(C_pageSize) + offset;
		checkInfEqual(result, pool + poolTop * uint64(C_pageSize));
		return result;
#else
        checkf(ptr < C_totalPages * C_pageSize, "Address: %u; Level: %u; %s:%u", ptr, level, debugFile, debugLine);
		return &pool[ptr];
#endif
	}
	HOST uint32* get_bucket_size_ptr(uint32 level, uint32 bucket) const
	{
        return &cpuData.bucketsSizes[HashDagUtils::get_bucket_global_index(level, bucket)];
    }
	HOST uint32 get_bucket_size(uint32 level, uint32 bucket) const
	{
        return *get_bucket_size_ptr(level, bucket);
	}
#if ADD_FULL_NODES_FIRST
    HOST uint32* get_full_node_index_ptr(uint32 level)
    {
        check(level < MAX_LEVELS);
        check(cpuData.fullNodeIndices);
        return &cpuData.fullNodeIndices[level];
    }
    HOST uint32 get_full_node_index(uint32 level)
    {
        return *get_full_node_index_ptr(level);
    }
#endif

public:
    HOST uint32 find_leaf_node_in_bucket(uint32 bucketSize, uint32 inLevel, uint32 bucket, uint64 nodeBits BLOOM_FILTER_ARG(BloomFilter const& aQuery)) const
    {
        PROFILE_FUNCTION_SLOW();

        constexpr uint32 level = C_leafLevel;
        checkEqual(level, inLevel);

        // Iterate over the pages
        auto const baseVirtualPtr = HashDagUtils::make_ptr(level, bucket, 0);
#if ALTERNATE_NODEFN_LEAFPAGE_PREFETCH & 1
        {
            auto const page = HashDagUtils::get_page(baseVirtualPtr);
            prefetch_page( page );
        }
#endif // ~ ALTERNATE_NODEFN_LEAFPAGE_PREFETCH

        for (uint32 pindex = 0; pindex < bucketSize; pindex += C_pageSize)
        {
            auto const pageVirtualPtr = baseVirtualPtr + pindex;
            auto const page = HashDagUtils::get_page(pageVirtualPtr);

#if USE_BLOOM_FILTER
            if (!bloom_filter_query(page, aQuery))
            {
#if ENABLE_CHECKS
                uint32 const* sysPtr = get_sys_ptr(level, pageVirtualPtr);
                uint32 const eind = std::min(bucketSize, pindex + C_pageSize);
                for (uint32 index = pindex; index < eind; index += 2, sysPtr += 2)
                {
                    check(sysPtr == get_sys_ptr(level, HashDagUtils::make_ptr(level, bucket, index)));

                    uint64 bits;
                    std::memcpy(&bits, sysPtr, sizeof(uint64));

                    check(bits != nodeBits)
                }
#endif
                continue;
            }
#endif // ~ USE_BLOOM_FILTER

            uint32 const* sysPtr = get_sys_ptr(level, pageVirtualPtr);

#if BLOOM_FILTER_PREFETCH
            bloom_filter_prefetch(page + 1);
#endif // ~ BLOOM_FILTER_PREFETCH
#if ALTERNATE_NODEFN_LEAFPAGE_PREFETCH & 2
            prefetch_page(page + 1);
#endif // ~ ALTERNATE_NODEFN_LEAFPAGE_PREFETCH

#if ALTERNATE_NODEFN_AVX_LEAFSEARCH
            if (pindex + C_pageSize <= bucketSize)
            {
                auto const search = leafsearch_avx(nodeBits, sysPtr);
                if (0xFFFFFFFF != search)
                    return baseVirtualPtr + pindex + search;
            }
            else
            {
                uint32 const eind = bucketSize;
#else
            {
                uint32 const eind = std::min(bucketSize, pindex + C_pageSize);
#endif
                for (uint32 index = pindex; index < eind; index += 2, sysPtr += 2)
                {
                    check(sysPtr == get_sys_ptr(level, HashDagUtils::make_ptr(level, bucket, index)));

                    uint64 bits;
                    std::memcpy(&bits, sysPtr, sizeof(uint64));

                    if (bits == nodeBits)
                        return baseVirtualPtr + index;
                }
            }
        }

        return 0xFFFFFFFF;
	}

    HOST uint32 find_interior_node_in_bucket(
            const uint32 bucketSize,
            const uint32 level,
            const uint32 bucket,
            const uint32 nodeSize, const uint32* __restrict__ node
            BLOOM_FILTER_ARG(BloomFilter const& aQuery)) const
    {
        PROFILE_FUNCTION_SLOW();

        check(Utils::total_size(node[0]) > 0);

        // Iterate over the pages
        auto const baseVirtualPtr = HashDagUtils::make_ptr(level, bucket, 0);
#if ALTERNATE_NODEFN_INTPAGE_PREFETCH & 1
        {
            auto const page = HashDagUtils::get_page(baseVirtualPtr);
            prefetch_page( page );
        }
#endif // ~ ALTERNATE_NODEFN_INTPAGE_PREFETCH

        for (uint32 pindex = 0; pindex < bucketSize; pindex += C_pageSize)
        {
            auto const pageVirtualPtr = baseVirtualPtr + pindex;
            auto const page = HashDagUtils::get_page(pageVirtualPtr);

#if USE_BLOOM_FILTER
            if (!bloom_filter_query(page, aQuery))
            {
#if ENABLE_CHECKS
                uint32 const* sysPtr = get_sys_ptr(level, pageVirtualPtr);

                uint32 pageEndIndex = std::min(bucketSize, pindex + C_pageSize);
                if (pindex + nodeSize < pageEndIndex)
                {
                    pageEndIndex -= nodeSize;

                    for (uint32 index = pindex; index < pageEndIndex;)
                    {
                        check(sysPtr == get_sys_ptr(level, HashDagUtils::make_ptr(level, bucket, index)));

                        check(0 != std::memcmp(node, sysPtr, nodeSize * sizeof(uint32)));

                        auto const nodeLength = Utils::total_size(*sysPtr);
                        index += nodeLength;
                        sysPtr += nodeLength;
                    }
                }
#endif
                continue;
            }
#endif // ~ USE_BLOOM_FILTER

            uint32 const* sysPtr = get_sys_ptr(level, pageVirtualPtr);

            uint32 pageEndIndex = std::min(bucketSize, pindex + C_pageSize);
            if (pindex + nodeSize >= pageEndIndex)
                return 0xFFFFFFFF;

            pageEndIndex -= nodeSize;

#if BLOOM_FILTER_PREFETCH
            bloom_filter_prefetch( page+1 );
#endif // ~ BLOOM_FILTER_PREFETCH
#if ALTERNATE_NODEFN_INTPAGE_PREFETCH & 2
            prefetch_page( page+1 );
#endif // ~ ALTERNATE_NODEFN_INTPAGE_PREFETCH

            for (uint32 index = pindex; index < pageEndIndex;)
            {
                check(sysPtr == get_sys_ptr(level, HashDagUtils::make_ptr(level, bucket, index)));

                if (0 == std::memcmp(node, sysPtr, nodeSize * sizeof(uint32)))
                    return baseVirtualPtr + index;

                auto const nodeLength = Utils::total_size(*sysPtr);
                index += nodeLength;
                sysPtr += nodeLength;
            }
        }

        return 0xFFFFFFFF;
    }

    HOST uint32 add_leaf_node(
            const uint32 inLevel,
            const uint64 leaf,
            const uint32 hash
            BLOOM_FILTER_ARG(BloomFilter const& aQuery))
    {
        PROFILE_FUNCTION_SLOW();

        static_assert(C_pageSize % 2 == 0, "add_leaf_node(): page size must be multiple of two");

        constexpr uint32 level = C_leafLevel;
        checkEqual(level, inLevel);

        auto const bucket = HashDagUtils::get_bucket_from_hash(level, hash);

        auto* const bucketSizePtr = get_bucket_size_ptr(level, bucket);
        auto bucketSize = *bucketSizePtr;

        uint32 const ptr = HashDagUtils::make_ptr(level, bucket, bucketSize);
        uint32 const page = HashDagUtils::get_page(ptr);

#if HAS_PAGE_TABLE
        // New Page?
        if (0 == (bucketSize % C_pageSize) && !is_page_allocated(page)) // Page is already allocated in case of GC
            allocate_page(page);

		check(is_page_allocated(page));
#endif

		// Append node
        std::memcpy(get_sys_ptr(level, ptr), &leaf, sizeof(uint64));

        // Edit the bucket size _after_ copying the node
        *bucketSizePtr = bucketSize + 2;
        checkInf(*bucketSizePtr, HashDagUtils::get_bucket_size(level));

#if USE_BLOOM_FILTER
        bloom_filter_insert(page, aQuery);
#endif // ~ USE_BLOOM_FILTER

        return ptr;
    }
    HOST uint32 add_interior_node(
            const uint32 level,
            const uint32 nodeSize, const uint32* __restrict__ node,
            const uint32 hash
            BLOOM_FILTER_ARG(BloomFilter const& aQuery))
    {
        PROFILE_FUNCTION_SLOW();

        check(nodeSize > 1);

        auto const bucket = HashDagUtils::get_bucket_from_hash(level, hash);

        auto* const bucketSizePtr = get_bucket_size_ptr(level, bucket);
        auto bucketSize = *bucketSizePtr;

#if HAS_PAGE_TABLE
        uint32 ptr, page;
        // Make sure the node is in a single page
        auto const pageSpaceLeft = C_pageSize - (bucketSize % C_pageSize);
        if (C_pageSize == pageSpaceLeft || pageSpaceLeft < nodeSize)
        {
            // Update bucket size to include the page boundary
            if (C_pageSize != pageSpaceLeft)
                bucketSize += pageSpaceLeft;

#if TRACK_PAGE_PADDING
            if (C_pageSize != pageSpaceLeft)
            {
                check(1 <= pageSpaceLeft && pageSpaceLeft < 9);
                cpuData.pagePaddingWastedMemory.fetch_add(pageSpaceLeft);
            }
#endif // ~ TRACK_PAGE_PADDING

            ptr = HashDagUtils::make_ptr(level, bucket, bucketSize);
            page = HashDagUtils::get_page(ptr);

            // New page, but still check if it's not allocated in case GC ran
            if (!is_page_allocated(page))
            {
                allocate_page(page);
            }
        }
        else
        {
            ptr = HashDagUtils::make_ptr(level, bucket, bucketSize);
            page = HashDagUtils::get_page(ptr);
        }
        check(is_page_allocated(page));
#else
        const uint32 ptr = HashDagUtils::make_ptr(level, bucket, bucketSize);
#endif

        uint32* __restrict__ sysPtr = get_sys_ptr(level, ptr);
		for (uint32 index = 0; index < nodeSize; index++)
		{
			sysPtr[index] = node[index];
		}

        // Edit the bucket size _after_ copying the node
        *bucketSizePtr = bucketSize + nodeSize;
		checkf(*bucketSizePtr < HashDagUtils::get_bucket_size(level), "Bucket size on level %u too low! Current size: %u; Required: >%u\n", level, HashDagUtils::get_bucket_size(level), *bucketSizePtr);

#if USE_BLOOM_FILTER
        bloom_filter_insert(page, aQuery);
#endif // ~ USE_BLOOM_FILTER

        return ptr;
    }

	HOST uint32 find_or_add_leaf_node(
	        const uint32 inLevel,
	        const uint64 leaf)
	{
        PROFILE_FUNCTION_SLOW();

        constexpr uint32 level = C_leafLevel;
        checkEqual(level, inLevel);

        const uint32 hash = HashDagUtils::hash_leaf(leaf);
        const uint32 bucket = HashDagUtils::get_bucket_from_hash(level, hash);
        const uint32 bucketSize = get_bucket_size(level, bucket);

#if USE_BLOOM_FILTER
		BloomFilter query;
        bloom_filter_init_leaf(query, leaf);
#endif // ~ USE_BLOOM_FILTER

		uint32 result = find_leaf_node_in_bucket(bucketSize, level, bucket, leaf BLOOM_FILTER_ARG(query));
        if (result == 0xFFFFFFFF)
        {
#if ENABLE_THREAD_SAFE_HASHTABLE
            ScopeLock lock(level, bucket);
            // Find again in case something changed
            const uint32 newBucketSize = get_bucket_size(level, bucket);
            if (newBucketSize != bucketSize)
            {
                result = find_leaf_node_in_bucket(newBucketSize, level, bucket, leaf BLOOM_FILTER_ARG(query));
            }
            if (result == 0xFFFFFFFF)
#endif // ~ ENABLE_THREAD_SAFE_HASHTABLE
            {
                result = add_leaf_node(level, leaf, hash BLOOM_FILTER_ARG(query));
            }
		}

#if ENABLE_CHECKS
        {
            const uint32* nodePtr = get_sys_ptr(level, result);
            check(nodePtr[0] == uint32(leaf));
            check(nodePtr[1] == uint32(leaf >> 32));
        }
#endif
		return result;
	}
	HOST uint32 find_or_add_interior_node(
	        const uint32 level,
	        const uint32 nodeSize, const uint32* __restrict__ node)
	{
        PROFILE_FUNCTION_SLOW();

        check(nodeSize > 1);

        const uint32 hash = HashDagUtils::hash_interior(nodeSize, node);
        const uint32 bucket = HashDagUtils::get_bucket_from_hash(level, hash);
        const uint32 bucketSize = get_bucket_size(level, bucket);

#if USE_BLOOM_FILTER
        BloomFilter query;
        bloom_filter_init_interior(query, nodeSize, node);
#endif

		uint32 result = find_interior_node_in_bucket(bucketSize, level, bucket, nodeSize, node BLOOM_FILTER_ARG(query));
		if (result == 0xFFFFFFFF)
		{
#if ENABLE_THREAD_SAFE_HASHTABLE
            ScopeLock lock(level, bucket);
            // Find again in case something changed
            const uint32 newBucketSize = get_bucket_size(level, bucket);
            if (newBucketSize != bucketSize)
            {
                result = find_interior_node_in_bucket(newBucketSize, level, bucket, nodeSize, node BLOOM_FILTER_ARG(query));
            }
            if (result == 0xFFFFFFFF)
#endif
			{
				result = add_interior_node(level, nodeSize, node, hash BLOOM_FILTER_ARG(query));
			}
		}

#if ENABLE_CHECKS
		{
            const uint32* nodePtr = get_sys_ptr(level, result);
            for (uint32 index = 0; index < nodeSize; index++)
            {
                check(nodePtr[index] == node[index]);
            }
		}
#endif
		return result;
	}

public:
#if USE_BLOOM_FILTER
    HOST static void bloom_filter_init_interior(
            BloomFilter& aQuery,
            const uint32 nodeSize, const uint32*__restrict__ node)
    {
        PROFILE_FUNCTION_SLOW();
    	
        auto const h64 = HashDagUtils::bloom_hash_interior(nodeSize, node);
        auto const a = uint32(h64 >> 32), b = uint32(h64);

        std::memset(aQuery.data(), 0, sizeof(uint32) * BLOOM_FILTER_WORDS);
        for (uint32 i = 0; i < BLOOM_FILTER_HASHES; ++i)
        {
            auto const idx = Utils::nth_hash(i, a, b, 32 * BLOOM_FILTER_WORDS);
            aQuery[idx / 32] |= (1u << (idx % 32));
        }
    }
    HOST static void bloom_filter_init_leaf(BloomFilter& aQuery, uint64 aLeaf)
    {
        PROFILE_FUNCTION_SLOW();
    	
        auto const h64 = HashDagUtils::bloom_hash_leaf(aLeaf);
        auto const a = uint32(h64 >> 32), b = uint32(h64);

        std::memset(aQuery.data(), 0, sizeof(uint32) * BLOOM_FILTER_WORDS);
        for (uint32 i = 0; i < BLOOM_FILTER_HASHES; ++i)
        {
            auto const idx = Utils::nth_hash(i, a, b, 32 * BLOOM_FILTER_WORDS);
            aQuery[idx / 32] |= (1u << (idx % 32));
        }
    }

    HOST bool bloom_filter_query(uint32 aPage, BloomFilter const& aQuery) const
    {
        PROFILE_FUNCTION_SLOW();
    	
#if BLOOM_FILTER_STATS
        ++cpuData.sBloomQueryCount;
#endif

#		if !BLOOM_FILTER_AVX
        auto const& __restrict__ filter = cpuData.mBloomPool[pageTable[aPage]];
        for (std::size_t i = 0; i < aQuery.size(); ++i)
        {
            if ((aQuery[i] & filter[i]) != aQuery[i])
                return false;
        }
#		else // BLOOM_FILTER_AVX
		auto const* qptr = reinterpret_cast<__m256i const*>(aQuery.data());

        auto const* filter = cpuData.mBloomPool+pageTable[aPage];
		auto const* fptr = reinterpret_cast<__m256i const*>(filter);
        for (std::size_t i = 0; i < BLOOM_FILTER_WORDS/8; ++i, ++qptr, ++fptr )
        {
			auto const q = _mm256_loadu_si256( qptr );
			auto const f = _mm256_loadu_si256( fptr );
			auto const masked = _mm256_and_si256( q, f );
			auto const comp = _mm256_cmpeq_epi64( q, masked );
			if( _mm256_movemask_epi8( comp ) != -1 )
				return false;
        }
#		endif // ~ BLOOM_FILTER_AVX

#if BLOOM_FILTER_STATS
        ++cpuData.sBloomHitCount;
#endif

        return true;
    }
    HOST void bloom_filter_insert(uint32 aPage, BloomFilter const& aNode)
    {
        checkInf(aPage, C_totalPages);
        auto& __restrict__ filter = cpuData.mBloomPool[pageTable[aPage]];
        for (std::size_t i = 0; i < aNode.size(); ++i)
            filter[i] |= aNode[i];
    }
    HOST void bloom_filter_reset(uint32 aPage)
    {
        checkInf(aPage, C_totalPages);
        auto& __restrict__ filter = cpuData.mBloomPool[pageTable[aPage]];
        for (std::size_t i = 0; i < BLOOM_FILTER_WORDS; ++i)
            filter[i] = 0;
    }

#if BLOOM_FILTER_PREFETCH
	HOST void bloom_filter_prefetch( uint32 aPage ) const
	{
		// XXX-investigate: should we do the if(), or just prefetch page
		// zero? Either is valid (but have different kinds of overheads)
		if( auto const pindex = pageTable[aPage] )
		{
			auto const* addr = cpuData.mBloomPool + pindex;
			Utils::prefetch_ro( addr );
		}
	}
#endif // ~ BLOOM_FILTER_PREFETCH
#endif

#if ALTERNATE_NODEFN_LEAFPAGE_PREFETCH || ALTERNATE_NODEFN_INTPAGE_PREFETCH
	HOST void prefetch_page( uint32 aPage ) const
	{
		if( auto const pindex = pageTable[aPage] )
		{
			auto const* addr = cpuData.cpuPool + pindex;
			Utils::prefetch_ro( addr );
		}
	}
#endif // ~ ALTERNATE_NODEFN_LEAFPAGE_PREFETCH
#if ALTERNATE_NODEFN_AVX_LEAFSEARCH
	HOST uint32 leafsearch_avx( uint64 aLeafBits, uint32 const* aLeafPagePtr ) const
	{
		// Broadcast aLeafBits into a __m256i register.
		auto const needle = _mm256_set1_epi64x( aLeafBits );

		auto const iter = sizeof(uint32)*std::size_t(C_pageSize) / sizeof(__m256i);
		auto const* beg = reinterpret_cast<__m256i const*>(aLeafPagePtr);
		auto const* end = beg + iter;
		for( ; beg != end; ++beg )
		{
			auto const haystack = _mm256_loadu_si256( beg );
			auto const wideMask = _mm256_cmpeq_epi64( needle, haystack );
			auto const mask = _mm256_movemask_epi8( wideMask );
			if( 0 != mask )
				return __tzcnt_u32( mask ) / 8;
		}

		return 0xFFFFFFFF;
	}
#endif // ~ ALTERNATE_NODEFN_AVX_LEAFSEARCH

public:
	HOST void print_stats() const
	{
		printf("##############################################\n");
		printf("total virtual used size: %fMB\n", get_virtual_used_size(true));
		printf("total allocated pages size: %fMB\n", get_allocated_pages_size());
		printf("total pool size: %fMB\n", get_pool_size());
		printf("page table size: %fMB\n", get_page_table_size());
        printf("page padding wasted memory: %fMB\n", Utils::to_MB(cpuData.pagePaddingWastedMemory.load()));
		printf("##############################################\n");
	}
	HOST double get_allocated_pages_size() const
	{
#if HAS_PAGE_TABLE
		return Utils::to_MB((poolTop - 1) * C_pageSize * sizeof(uint32));
#else
    return 0;
#endif
	}
	HOST double get_pool_size() const
	{
#if HAS_PAGE_TABLE
		return Utils::to_MB(cpuData.poolMaxSize * C_pageSize * sizeof(uint32));
#else
    return C_totalPages * C_pageSize;
#endif
	}
	HOST double get_page_table_size() const
	{
#if HAS_PAGE_TABLE
		return Utils::to_MB(pageTableSize * sizeof(uint32));
#else
    return 0;
#endif
	}

	HOST size_t get_total_pages() const
	{
#if HAS_PAGE_TABLE
		return cpuData.poolMaxSize;
#else
    	return 0;
#endif
	}
	HOST size_t get_allocated_pages() const
	{
#if HAS_PAGE_TABLE
		return poolTop - 1;
#else
    	return 0;
#endif
	}


public:
#if ADD_FULL_NODES_FIRST
	HOST void add_full_node(uint32 level)
    {
        check(get_bucket_size(level, 0) == 0);

        if (level == C_leafLevel)
        {
            const uint64 leaf = uint64(-1);
            const uint32 hash = HashDagUtils::hash_leaf(leaf);

#if USE_BLOOM_FILTER
            BloomFilter filter;
            bloom_filter_init_leaf(filter, leaf);
#endif // ~ USE_BLOOM_FILTER

            cpuData.fullNodeIndices[level] = add_leaf_node(C_leafLevel, leaf, hash BLOOM_FILTER_ARG(filter));
        }
        else
        {
            const uint32 numVoxelsInFullNode =
                    level < C_colorTreeLevels
                    ? 0
                    : (1u << (3 * (C_maxNumberOfLevels - level))); // 8^depth

            checkfAlways(numVoxelsInFullNode < (1u << 24), "need to increase C_colorTreeLevels");

            uint32 nodeBuffer[9];
            const uint32 nodeBufferSize = 9;
            nodeBuffer[0] = (numVoxelsInFullNode << 8) | 0xFF;
            for (uint32 child = 0; child < 8; child++)
                nodeBuffer[1 + child] = cpuData.fullNodeIndices[level + 1];

            const uint32 hash = HashDagUtils::hash_interior(nodeBufferSize, nodeBuffer);

#if USE_BLOOM_FILTER
            BloomFilter newFilter;
            bloom_filter_init_interior(newFilter, nodeBufferSize, nodeBuffer);
#endif // ~ USE_BLOOM_FILTER

            cpuData.fullNodeIndices[level] = add_interior_node(level, nodeBufferSize, nodeBuffer, hash BLOOM_FILTER_ARG(newFilter));
        }
    }
#endif

private:
#if HAS_PAGE_TABLE
	HOST void allocate_page(uint32 page)
	{
#if ENABLE_THREAD_SAFE_HASHTABLE
		PageScopeLock lock;
#endif
		check(pageTable && pool);
		check(page < pageTableSize);
		check(pageTable[page] == 0);
		pageTable[page] = poolTop++;
		check(poolTop <= cpuData.poolMaxSize);
#if 0
		printf("allocating page %d\n", poolTop);
#endif

		check(is_page_allocated(page));
	}
	HOST_DEVICE bool is_page_allocated(uint32 page) const
	{
		check(pageTable && pool);
		check(page < pageTableSize);
		return pageTable[page] != 0;
	}
#endif

private:
    // Keep those 2 on the GPU too for checks
    uint32 pageTableSize = 0;
    uint32 poolTop = 1; // In number of pages. Start at 1 to be able to have pages set to 0. Means we're wasting pageSize memory, but that's ok
#if MANUAL_VIRTUAL_MEMORY
    uint32* __restrict__ gpuPageTable = nullptr;
#endif

	uint32* __restrict__ gpuPool = nullptr;

private:
    struct CpuData
    {
    public:
        uint32* __restrict__ bucketsSizes = nullptr;
#if MANUAL_CPU_DATA
        uint32* __restrict__ lastBucketsSizes = nullptr;
        uint32* __restrict__ cpuPageTable = nullptr;
        uint32* __restrict__ cpuPool = nullptr;
#endif
#if ADD_FULL_NODES_FIRST
        uint32* __restrict__ fullNodeIndices = nullptr;
#endif
#if USE_BLOOM_FILTER
        BloomFilter* __restrict__ mBloomPool = nullptr;
#endif
    public:
	    uint32 poolMaxSize = 0; // In number of pages

    public:
        uint8 padding0[128]; // we don't want these on the same cache line
#if ENABLE_THREAD_SAFE_HASHTABLE
        std::mutex mutexes[C_totalNumberOfBuckets];
        std::mutex pageMutex;
#endif
        uint8 padding1[128];
        std::atomic<uint32> pagePaddingWastedMemory{ 0 }; // in number of uint32
#if BLOOM_FILTER_STATS
        uint8 padding2[128];
        std::atomic<uint64> sBloomQueryCount{ 0 };
        uint8 padding3[128];
        std::atomic<uint64> sBloomHitCount{ 0 };
#endif
    };
    static CpuData cpuData;

private:
#if ENABLE_THREAD_SAFE_HASHTABLE
	struct ScopeLock
	{
	public:
		ScopeLock(uint32 level, uint32 bucket)
			: mutex(cpuData.mutexes[HashDagUtils::get_bucket_global_index(level, bucket)])
		{
			mutex.lock();
		}
		~ScopeLock()
		{
			mutex.unlock();
		}

	private:
		std::mutex& mutex;
	};

	struct PageScopeLock
	{
	public:
		PageScopeLock()
		{
			cpuData.pageMutex.lock();
		}
		~PageScopeLock()
		{
			cpuData.pageMutex.unlock();
		}
	};
#endif

	friend struct HashDAGFactory;
};

#undef pageTable
#undef pool
