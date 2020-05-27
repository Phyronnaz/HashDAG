#include "hash_table.h"
#include "cuda_error_check.h"
#include "memory.h"

void HashTable::create(uint32 poolSizeInPages)
{
    PROFILE_FUNCTION();

	check(!gpuPool);

	const auto prefixString = Stats::get_prefix();
	const auto printPrefix = prefixString.c_str();

	cpuData.poolMaxSize = poolSizeInPages;
	pageTableSize = C_totalPages;

#if MANUAL_CPU_DATA
	printf("%sCreating pool with %d virtual pages and %d physical pages\n", printPrefix, pageTableSize, poolSizeInPages);
	printf("%s pool size: %fMB\n", printPrefix, get_pool_size());
	printf("%s page table size: %fMB\n", printPrefix, get_page_table_size());

    cpuData.cpuPool = Memory::malloc<uint32>("CPU pool", sizeof(uint32) * poolSizeInPages * C_pageSize, EMemoryType::CPU);
    cpuData.cpuPageTable = Memory::malloc<uint32>("CPU page table", sizeof(uint32) * pageTableSize, EMemoryType::CPU);

	for (uint32 page = 0; page < pageTableSize; page++)
	{
		cpuData.cpuPageTable[page] = 0;
	}
#endif

    cpuData.bucketsSizes = Memory::malloc<uint32>("bucket sizes", C_totalNumberOfBuckets * sizeof(uint32), EMemoryType::CPU);
#if MANUAL_CPU_DATA
    cpuData.lastBucketsSizes = Memory::malloc<uint32>("last bucket sizes", C_totalNumberOfBuckets * sizeof(uint32), EMemoryType::CPU);
#endif

    for (uint32 bucket = 0; bucket < C_totalNumberOfBuckets; bucket++)
    {
        cpuData.bucketsSizes[bucket] = 0;
#if MANUAL_CPU_DATA
        cpuData.lastBucketsSizes[bucket] = 0;
#endif
    }

#if USE_BLOOM_FILTER
    printf("%s bloom filter pool: %.1fMB\n", printPrefix, Utils::to_MB(sizeof(BloomFilter) * poolSizeInPages));
    cpuData.mBloomPool = Memory::malloc<BloomFilter>("bloom filter", poolSizeInPages * sizeof(BloomFilter), EMemoryType::CPU);
    std::memset(cpuData.mBloomPool, 0, poolSizeInPages * sizeof(BloomFilter));
#endif // ~ USE_BLOOM_FILTER

#if MANUAL_VIRTUAL_MEMORY
    const auto type = MANUAL_CPU_DATA ? EMemoryType::GPU_Malloc : EMemoryType::GPU_Managed;
	gpuPool = Memory::malloc<uint32>("hash map pool", poolSizeInPages * C_pageSize * sizeof(uint32), type);
	gpuPageTable = Memory::malloc<uint32>("hash map page table", pageTableSize * sizeof(uint32), type);

#if !MANUAL_CPU_DATA
	for (uint32 page = 0; page < pageTableSize; page++)
	{
		gpuPageTable[page] = 0;
	}
#endif
#else
    printf("%sCreating GPU pool with %fMB allocated memory (%d pages)\n", printPrefix, Utils::to_MB(C_totalPages * C_pageSize * sizeof(uint32)), C_totalPages);
    gpuPool = Memory::malloc<uint32>("hash map pool", C_totalPages * C_pageSize * sizeof(uint32));
#endif

#if 0 // no performance improvement
	CUDA_CHECKED_CALL cudaMemAdvise(pool, C_pageSize * poolMaxSize * sizeof(uint32), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
	CUDA_CHECKED_CALL cudaMemAdvise(pageTable, pageTableSize * sizeof(uint32), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
	CUDA_CHECKED_CALL cudaMemAdvise(bucketsSizes, C_totalNumberOfBuckets * sizeof(uint32), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
#endif
}

void HashTable::destroy()
{
    PROFILE_FUNCTION();
	
    check(gpuPool);

	Memory::free(gpuPool);
	gpuPool = nullptr;

#if MANUAL_VIRTUAL_MEMORY
    Memory::free(gpuPageTable);
    gpuPageTable = nullptr;
#endif

#if MANUAL_CPU_DATA
    Memory::free(cpuData.cpuPool);
    cpuData.cpuPool = nullptr;

    Memory::free(cpuData.cpuPageTable);
    cpuData.cpuPageTable = nullptr;

    Memory::free(cpuData.lastBucketsSizes);
    cpuData.lastBucketsSizes = nullptr;
#endif

    Memory::free(cpuData.bucketsSizes);
    cpuData.bucketsSizes = nullptr;

#if USE_BLOOM_FILTER
    Memory::free(cpuData.mBloomPool);
    cpuData.mBloomPool = nullptr;
#endif

#if BLOOM_FILTER_STATS
	printf("Bloom filter: %zu queries, %zu 'hits'\n", std::size_t(cpuData.sBloomQueryCount.load()), std::size_t(cpuData.sBloomHitCount.load()) );
#endif
}

void HashTable::prefetch()
{
#if 0 // No performance improvement
	CUDA_CHECKED_CALL cudaMemPrefetchAsync(pool, C_pageSize * poolMaxSize * sizeof(uint32), cudaCpuDeviceId);
	CUDA_CHECKED_CALL cudaMemPrefetchAsync(pageTable, pageTableSize * sizeof(uint32), cudaCpuDeviceId);
	CUDA_CHECKED_CALL cudaMemPrefetchAsync(bucketsSizes, C_totalNumberOfBuckets * sizeof(uint32), cudaCpuDeviceId);
#endif
}

void HashTable::upload_to_gpu()
{
    PROFILE_FUNCTION();
	
#if MANUAL_CPU_DATA
#if MANUAL_VIRTUAL_MEMORY
    CUDA_CHECKED_CALL cudaMemcpyAsync(gpuPageTable, cpuData.cpuPageTable, pageTableSize * sizeof(uint32), cudaMemcpyHostToDevice);
#endif

    const auto copy_page_to_gpu = [&](uint32 page, uint32 start, uint32 size)
    {
        check(page < pageTableSize);
        check(is_page_allocated(page))
        check(start + size <= C_pageSize);
        const uint32 pageValue = cpuData.cpuPageTable[page];
        check(C_pageSize * pageValue + start + size <= cpuData.poolMaxSize * C_pageSize);
#if MANUAL_VIRTUAL_MEMORY
        const uint32 gpuPage = pageValue;
        CUDA_CHECKED_CALL cudaMemcpyAsync(gpuPool + C_pageSize * gpuPage + start, cpuData.cpuPool + C_pageSize * pageValue + start, size * sizeof(uint32), cudaMemcpyHostToDevice);
#else
        // Using cudaMemcpyAsync on managed memory seems to kill performance
        const uint32 gpuPage = page;
        std::memcpy(gpuPool + C_pageSize * gpuPage + start, cpuData.cpuPool + C_pageSize * pageValue + start, size * sizeof(uint32));
#endif
    };

    for (uint32 level = 0; level < C_maxNumberOfLevels; level++)
    {
        for (uint32 bucket = 0; bucket < HashDagUtils::get_buckets_per_level(level); bucket++)
        {
            const uint32 bucketIndex = HashDagUtils::get_bucket_global_index(level, bucket);
            const uint32 bucketSize = cpuData.bucketsSizes[bucketIndex];
            uint32& lastBucketSize = cpuData.lastBucketsSizes[bucketIndex];
            if (bucketSize != lastBucketSize)
            {
                checkInf(lastBucketSize, bucketSize);
                const uint32 start = HashDagUtils::make_ptr(level, bucket, lastBucketSize);
                const uint32 end = HashDagUtils::make_ptr(level, bucket, bucketSize);

#if 1 // more precise upload
                uint32 position = start;
                while (position < end)
                {
                    const uint32 numLeftInPage = C_pageSize - (position % C_pageSize);
                    const uint32 numToSyncInPage = std::min(numLeftInPage, end - position);
                    copy_page_to_gpu(position / C_pageSize, position % C_pageSize, numToSyncInPage);
                    position += numToSyncInPage;
                    check(position == end || (position % C_pageSize) == 0);
                }
                checkEqual(position, end);
#else
                for (uint32 page = start / C_pageSize; page < Utils::divide_ceil(end, C_pageSize); page++)
                {
                    copy_page_to_gpu(page, 0, C_pageSize);
                }
#endif

                lastBucketSize = bucketSize;
            }
        }
    }
#endif
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
}

void HashTable::full_upload_to_gpu()
{
    PROFILE_FUNCTION();
	
#if MANUAL_CPU_DATA
#if MANUAL_VIRTUAL_MEMORY
    for (uint32 bucket = 0; bucket < C_totalNumberOfBuckets; bucket++)
    {
        cpuData.lastBucketsSizes[bucket] = cpuData.bucketsSizes[bucket];
    }
    Memory::cuda_memcpy(gpuPageTable, cpuData.cpuPageTable, pageTableSize * sizeof(uint32), cudaMemcpyHostToDevice);
    Memory::cuda_memcpy(gpuPool, cpuData.cpuPool, cpuData.poolMaxSize * C_pageSize * sizeof(uint32), cudaMemcpyHostToDevice);
#else
    for (uint32 bucket = 0; bucket < C_totalNumberOfBuckets; bucket++)
    {
        cpuData.lastBucketsSizes[bucket] = 0;
    }
    upload_to_gpu();
#endif
#endif
}
void HashTable::save_bucket_sizes(bool beforeEdits) const
{
    PROFILE_FUNCTION();
	
    printf("Dumping bucket sizes... ");
    std::stringstream path;
#ifdef PROFILING_PATH
        path << PROFILING_PATH << "/";
        path << STATS_FILES_PREFIX;
#else
		path << "./profiling/";
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
        path << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
#endif
    path << ".buckets";
    if (beforeEdits)
        path << ".before";
    else
        path << ".after";
    path << ".csv";

    std::ofstream os(path.str());
    checkAlways(os.is_open());

    for (uint32 level = 0; level < C_maxNumberOfLevels; level++)
    {
        os << "level;" << level << std::endl;
        for (uint32 bucket = 0; bucket < HashDagUtils::get_buckets_per_level(level); bucket++)
        {
            os << bucket << ";" << *get_bucket_size_ptr(level, bucket) << std::endl;
        }
    }

    checkAlways(os.good());
    os.close();
    printf("Done!\n");
}

double HashTable::get_virtual_used_size(bool printPerLevelUsage) const
{
	PROFILE_FUNCTION();
    
	uint32 totalVirtualSize = 0;
    for (uint32 level = 0; level < C_maxNumberOfLevels; level++)
    {
        uint32 virtualSize = 0;
        uint32 pagesCount = 0;
        for (uint32 bucket = 0; bucket < HashDagUtils::get_buckets_per_level(level); bucket++)
        {
            uint32 bucketSize = get_bucket_size(level, bucket);
            if (bucketSize > 0)
            {
                virtualSize += uint32(bucketSize * sizeof(uint32));
                pagesCount += Utils::divide_ceil(bucketSize, C_pageSize);
            }
        }
        totalVirtualSize += virtualSize;
        if (printPerLevelUsage)
        {
            printf("level %u: \n\tvirtual size: %fMB \n\treal size: %fMB \n\tpages: %u\n",
                   level,
                   Utils::to_MB(virtualSize),
                   Utils::to_MB(pagesCount * C_pageSize * sizeof(uint32)),
                   pagesCount);
        }
    }
    return Utils::to_MB(totalVirtualSize);
}

uint32 HashTable::get_level_size(uint32 level) const
{
    uint32 count = 0;
    for (uint32 bucket = 0; bucket < HashDagUtils::get_buckets_per_level(level); bucket++)
    {
        count += get_bucket_size(level, bucket);
    }
    return count;
}

HashTable::CpuData HashTable::cpuData;
