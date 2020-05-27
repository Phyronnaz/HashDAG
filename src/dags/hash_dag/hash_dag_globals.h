#pragma once

#include "typedefs.h"
#include <limits>
#include "utils.h"

constexpr uint32 C_colorTreeLevels = 10;

// in number of uint32
constexpr uint32 C_pageSize = PAGE_SIZE;
constexpr uint32 C_maxNumberOfLevels = MAX_LEVELS;
constexpr uint32 C_leafLevel = C_maxNumberOfLevels - 2;

constexpr uint32 C_bucketSizeForTopLevels = BUCKETS_SIZE_FOR_TOP_LEVELS;
constexpr uint32 C_bucketSizeForLowLevels = BUCKETS_SIZE_FOR_LOW_LEVELS;
constexpr uint32 C_pagesPerBucketForTopLevels = C_bucketSizeForTopLevels / C_pageSize;
constexpr uint32 C_pagesPerBucketForLowLevels = C_bucketSizeForLowLevels / C_pageSize;

constexpr uint32 C_bucketsPerTopLevel = 1 << BUCKETS_BITS_FOR_TOP_LEVELS;
constexpr uint32 C_bucketsPerLowLevel = 1 << BUCKETS_BITS_FOR_LOW_LEVELS;
constexpr uint32 C_bucketsNumTopLevels = BUCKETS_NUM_TOP_LEVELS;

constexpr uint32 C_bucketsNumLowLevels = C_maxNumberOfLevels - C_bucketsNumTopLevels;

constexpr uint32 C_totalNumberOfBucketsInTopLevels = C_bucketsNumTopLevels * C_bucketsPerTopLevel;
constexpr uint32 C_totalNumberOfBucketsInLowLevels = C_bucketsNumLowLevels * C_bucketsPerLowLevel;
constexpr uint32 C_totalNumberOfBuckets = C_totalNumberOfBucketsInTopLevels + C_totalNumberOfBucketsInLowLevels;

constexpr uint32 C_totalPages = C_totalNumberOfBucketsInTopLevels * C_pagesPerBucketForTopLevels + C_totalNumberOfBucketsInLowLevels * C_pagesPerBucketForLowLevels;
constexpr uint32 C_totalVirtualAddresses = C_totalPages * C_pageSize;

static_assert(C_bucketSizeForTopLevels % C_pageSize == 0, "");
static_assert(C_bucketSizeForLowLevels % C_pageSize == 0, "");
static_assert((C_pageSize & (C_pageSize - 1)) == 0, "");
static_assert((C_bucketSizeForTopLevels & (C_bucketSizeForTopLevels - 1)) == 0, "");
//static_assert((C_bucketSizeForLowLevels & (C_bucketSizeForLowLevels - 1)) == 0, ""); //XXX-probably too restrictive?
static_assert((C_bucketsPerTopLevel & (C_bucketsPerTopLevel - 1)) == 0, "");
static_assert((C_bucketsPerLowLevel & (C_bucketsPerLowLevel - 1)) == 0, "");
static_assert(uint64(C_totalPages) * uint64(C_pageSize) < std::numeric_limits<uint32>::max(), "virtual address space too big");
static_assert(uint64(C_totalNumberOfBucketsInTopLevels) * uint64(C_pagesPerBucketForTopLevels) + uint64(C_totalNumberOfBucketsInLowLevels) + uint64(C_pagesPerBucketForLowLevels) < std::numeric_limits<uint32>::max(), "virtual address space really too big");
