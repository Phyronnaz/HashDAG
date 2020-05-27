#pragma once

#include "script_definitions.h"

/** Config: If true, will disable all the stuff we don't want when recording benchmarks
 */
#ifndef BENCHMARK
#define BENCHMARK 0
#endif

/** Config: Will replay a video
 */
#ifndef USE_VIDEO
#define USE_VIDEO 0
#endif

/** Config: setup stuff for video replay recording
 */
#ifndef RECORD_VIDEO
#define RECORD_VIDEO 0
#endif

#if USE_VIDEO
#define SCENE_DEPTH 17
#define REPLAY_DEPTH 17
#define ENABLE_CHECKS 0
#elif RECORD_VIDEO
#define SCENE_DEPTH 17
#define REPLAY_DEPTH 17
#define ENABLE_CHECKS 0
#define RECORD_TOOL_OVERLAY 1
#endif

/** Config: Enable all asserts
 */
#ifndef ENABLE_CHECKS
#define ENABLE_CHECKS (0 && !BENCHMARK)
#endif

/** Config: start several threads to do the edits
 */
#ifndef THREADED_EDITS
#define THREADED_EDITS (1 && !BENCHMARK)
#endif

/** Config: Will lock to add when find fails
 */
#ifndef ENABLE_THREAD_SAFE_HASHTABLE
#define ENABLE_THREAD_SAFE_HASHTABLE THREADED_EDITS
#endif

/** Config: number of threads to use in the thread pool. If 0 will start all tasks in one thread
 */
#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

/** Config: Enables undo redo
 *
 * Will record the first nodes of the DAG, and keep previous color leaves instead of deleting them
 */
#ifndef UNDO_REDO
#define UNDO_REDO (1 && !BENCHMARK)
#endif

/** Config: shows a red overlay for tools
 *
 * Adds a small RT cost so disable when benchmarking
 */
#ifndef TOOL_OVERLAY
#define TOOL_OVERLAY (1 && !BENCHMARK)
#endif

/** Config: Store a debug hash in CompressedColor for block debug
 */
#ifndef COLOR_DEBUG
#define COLOR_DEBUG (1 && !BENCHMARK)
#endif

/** Config: enable editing colors
 *
 * If off, will show yellow voxels where the color index is invalid
 */
#ifndef EDITS_ENABLE_COLORS
#define EDITS_ENABLE_COLORS 1
#endif

/** Config: adds the full nodes at the start of the hash map, and keep a reference to them
 */
#ifndef ADD_FULL_NODES_FIRST
#define ADD_FULL_NODES_FIRST 1
#endif

/** Config: check if a section is full and use the corresponding full node if it is
 */
#ifndef EARLY_FULL_CHECK
#define EARLY_FULL_CHECK 1
#endif

#if EARLY_FULL_CHECK && !ADD_FULL_NODES_FIRST
#error "EARLY_FULL_CHECK requires ADD_FULL_NODES_FIRST"
#endif

/** Config: if true, use a page table; if false, use cuda managed memory over subscription as virtual memory
 */
#ifndef MANUAL_VIRTUAL_MEMORY
#define MANUAL_VIRTUAL_MEMORY 1
#endif

/** Config: if true, manually duplicate the data on the cpu; if false, use cuda managed memory with the data on the GPU
 *
 * Managed memory tends to be quite slow for writing
 */
#ifndef MANUAL_CPU_DATA
#define MANUAL_CPU_DATA 1
#endif

/** Config: call cudaMemAdviseSetReadMostly on managed memory to duplicate the data to the GPU (makes reads on the CPU faster)
 */
#ifndef MEM_ADVISE_READ_MOSTLY
#define MEM_ADVISE_READ_MOSTLY 1
#endif

/** Config: record memory GC would free up on every frame
 */
#ifndef SIMULATE_GC
#define SIMULATE_GC 0
#endif

/** Config: replay twice, and only record the second one
 *
 * Useful when recording raytracing times, to make sure everything is loaded in VRAM
 */
#ifndef REPLAY_TWICE
#define REPLAY_TWICE 0
#endif

/** Config: name of the scene to use
 */
#ifndef SCENE
#define SCENE "epiccitadel"
#endif

/** Config: video name
 */
#ifndef VIDEO_NAME
#define VIDEO_NAME "basic"
#endif

/** Config: depth of the scene. Will use the file named SCENEXk with X = 1 << (SCENE_DEPTH - 10)
 */
#ifndef SCENE_DEPTH
#define SCENE_DEPTH 16
#endif

/** Config: depth of the scene in which the replay was recorded.
 */
#ifndef REPLAY_DEPTH
#define REPLAY_DEPTH 16
#endif

/** Config: suffix of the replay to use
 *
 * Loaded replay is replays/SCENE_REPLAY_NAME.csv++++
 */
#ifndef REPLAY_NAME
#define REPLAY_NAME "move"
#endif

/** Config: load uncompressed colors?
 */
#ifndef LOAD_UNCOMPRESSED_COLORS
#define LOAD_UNCOMPRESSED_COLORS 0
#endif

/** Config: load compressed colors?
 */
#ifndef LOAD_COMPRESSED_COLORS
#define LOAD_COMPRESSED_COLORS 1
#endif

/** Config: add always_inline to all (non-recursive) functions
 *
 * Improves perf a little bit
 * Having it on makes GDB crash when debugging
 */
#ifndef ENABLE_FORCEINLINE
#define ENABLE_FORCEINLINE BENCHMARK
#endif

/** Config: add flatten to all (non-recursive) functions
 *
 * Improves perf a little bit
 */
#ifndef ENABLE_FLATTEN
#define ENABLE_FLATTEN BENCHMARK
#endif

/** Config: page size
 */
#ifndef PAGE_SIZE
#define PAGE_SIZE 512
#endif

/** Config: number of "top levels"
 *
 * top vs low levels have different bucket count/sizes
 */
#ifndef BUCKETS_NUM_TOP_LEVELS
#define BUCKETS_NUM_TOP_LEVELS 9
#endif

/** Config: number of bucket bits for the top levels
 *
 * Bucket count = 1 << bits
 */
#ifndef BUCKETS_BITS_FOR_TOP_LEVELS
#define BUCKETS_BITS_FOR_TOP_LEVELS 10
#endif

/** Config: number of bucket bits for the low levels
 *
 * Bucket count = 1 << bits
 */
#ifndef BUCKETS_BITS_FOR_LOW_LEVELS
#define BUCKETS_BITS_FOR_LOW_LEVELS 16
#endif

/** Config: size of top level buckets, in number of uint32
 */
#ifndef BUCKETS_SIZE_FOR_TOP_LEVELS
#define BUCKETS_SIZE_FOR_TOP_LEVELS 1024
#endif

/** Config: size of low level buckets, in number of uint32
 */
#ifndef BUCKETS_SIZE_FOR_LOW_LEVELS
#define BUCKETS_SIZE_FOR_LOW_LEVELS 4096
#endif

/** Config: record times of each part of the edits functions
 */
#ifndef EDITS_PROFILING
#define EDITS_PROFILING 0
#endif

/** Config: count number of iterated voxels, recursive calls, ...
 */
#ifndef EDITS_COUNTERS
#define EDITS_COUNTERS !BENCHMARK
#endif

/**  Config: Use 64 bits enclosed leaves
 *
 * Setting this to true will use 64 bits enclosed leaves to store the top levels voxel counts
 */
#ifndef BIG_ENCLOSED_LEAVES
#define BIG_ENCLOSED_LEAVES 1
#endif

/** Config: Use intrinsics for popc
 *
 * x86_64 CPUs have single-instruction popc (population count) instructions.
 * According to Wikipedia: "Intel considers POPCNT as part of SSE4.2, and LZCNT
 * as part of BMI1. POPCNT has a separate CPUID flag; however, Intel uses AMD's
 * ABM flag to indicate LZCNT support (since LZCNT completes the ABM)". Either
 * way, anything newer than 2008 (Nahelem or Barcelona) is likely to support
 * it. 
 *
 * GCC exposes it as __builtin_popcount(). MSVC (currently not implemented) 
 * would use either __popcnt() (intrin.h) or _mm_popcnt_u64() (nmmintrin.h)
 *
 * Performance impact: minor (but tends towards a tiny bit faster)
 */
#ifndef USE_POPC_INTRINSICS
#define USE_POPC_INTRINSICS 1
#endif

/** Config: Call murmurhash more "properly"
 *
 * The original implementation is a bit weird. It manually loops over values,
 * bangs these into a iterated value, and passes that value through the hash
 * repeatedly. Wikipedia's implementation already iterates over a 32 bit array,
 * so why not just do that?
 *
 * Performance impact: unchanged
 */
#ifndef USE_ALTERNATE_HASH
#define USE_ALTERNATE_HASH 1
#endif

/** Config: Use Bloom Filter
 *
 * Use a Bloom filter per page. With the bloom filter, it becomes possible to
 * check if a certain node potentially exists within a page, without accessing
 * the page and searching for the node. However, this comes at the cost of
 * accessing the memory where the bloom filter resides.
 *
 * The bloom filters are stored in a separate pool, parallel to the page pool.
 * A single bloom filter adds at the moment 20/128 ≈ 16% memory overhead. If
 * used for non-leaf nodes only, it's possible to reduce this to 8/128 ≈ 6%.
 * 
 * Right now, for all nodes (including leaf-nodes), the Bloom filter allows
 * skipping of a page about 66% of the time. For non-leaf nodes only, the rate
 * is much higher (but the overall amount of queries is much lower as well).
 *
 * Performance impact: neutral? :-(
 */
#ifndef USE_BLOOM_FILTER
#define USE_BLOOM_FILTER 1
#endif

#if USE_BLOOM_FILTER && (MANUAL_VIRTUAL_MEMORY && !MANUAL_CPU_DATA)
#	error "USE_BLOOM_FILTER: MANUAL_VIRTUAL_MEMORY and not MANUAL_CPU_DATA: not yet implemented"
#endif

/** Config: Use AVX2 intrinsics for computing and evaluating bloom filters
 */
#ifndef BLOOM_FILTER_AVX
#define BLOOM_FILTER_AVX 0
#endif

/* Config: Number of hashes
 *
 * Number of hashes that are inserted inserted into the Bloom filter. See
 * https://stackoverflow.com/a/22467497 for estimating this (and the size of
 * the bloom filter).
 */
#ifndef BLOOM_FILTER_HASHES
#define BLOOM_FILTER_HASHES 7
#endif 

/** Config: Number of words per Bloom filter
 *
 * Wikipedia suggests around 10 bits per element inserted into the Bloom
 * filter for a target p = 0.01 (1%):
 *   - For interior nodes: Assuming the average node is 4-5 words (1 childmask
 *     + 3-4 pointers), a 128 word page contains ~25.6 elements => total of 256
 *     bits with 10 bits per element. 256 bits => 8 words.
 *   - For leaf nodes: We have 2 words (64 bits) per page. For 128 word pages,
 *     this means a total of 640 bits => 20 words.
 *
 * Since we use the same size for both interior nodes and leaf nodes, we have
 * to go with the larger 20 word size.
 *
 * NOTE: needs to increase with page size too
 */
#ifndef BLOOM_FILTER_WORDS
#if BLOOM_FILTER_AVX
#define BLOOM_FILTER_WORDS (24*(PAGE_SIZE/128))
#else
#define BLOOM_FILTER_WORDS (20*(PAGE_SIZE/128))
#endif
#endif


#if BLOOM_FILTER_AVX && (BLOOM_FILTER_WORDS % 8) != 0
#	error "BLOOM_FILTER_AVX: requires BLOOM_FILTER_WORDS to be a multiple of 8"
#endif

/** Config: Gather stats on the Bloom filter
 *
 * Gathers number of queries and number of hits and prints these on exit.
 * This is mostly useful for debugging the bloom filter.
 * 
 * (TODO-FIXME)
 */
#ifndef BLOOM_FILTER_STATS
#define BLOOM_FILTER_STATS 0
#endif

#if BLOOM_FILTER_STATS && !USE_BLOOM_FILTER
#error "BLOOM_FILTER_STATS requires USE_BLOOM_FILTER"
#endif

/** Config: Prefetch bloom filters
 *
 * When enabled, prefetches bloom filter for the next iteration, before
 * searching for the current one.
 */
#ifndef BLOOM_FILTER_PREFETCH
#define BLOOM_FILTER_PREFETCH 0
#endif

/** Config: Track amount of padding.
 */
#ifndef TRACK_PAGE_PADDING
#define TRACK_PAGE_PADDING 1
#endif

/** Config: Intialize unallocated page-memory to invalid
 
 * TODO: not yet implemented. In theory not required.
 */
#ifndef ALTERNATE_NODEFN_INVALID_INIT //TODO
#define ALTERNATE_NODEFN_INVALID_INIT 0
#endif

/** Config: Prefetch leaf pages when searching
 *
 * Four settings:
 *   - 0 : no prefetching
 *   - 1 : prefetch the first page before evaluating the Bloom filter
 *   - 2 : prefetch next page before searching the current page
 *   - 3 : both 1 and 2
 */
#ifndef ALTERNATE_NODEFN_LEAFPAGE_PREFETCH
#define ALTERNATE_NODEFN_LEAFPAGE_PREFETCH 0 /* one of 0, 1, 2, 3 */
#endif

/** Config: Prefetch interior pages when searching
 *
 * Four settings:
 *   - 0 : no prefetching
 *   - 1 : prefetch the first page before evaluating the Bloom filter
 *   - 2 : prefetch next page before searching the current page
 *   - 3 : both 1 and 2
 */
#ifndef ALTERNATE_NODEFN_INTPAGE_PREFETCH
#define ALTERNATE_NODEFN_INTPAGE_PREFETCH ALTERNATE_NODEFN_LEAFPAGE_PREFETCH
#endif

/** Config: Use AVX2 intrinsics for leaf searches
 */
#ifndef ALTERNATE_NODEFN_AVX_LEAFSEARCH
#define ALTERNATE_NODEFN_AVX_LEAFSEARCH 0
#endif

/** Config: shade each voxel face independently according to its normal
 */
#ifndef PER_VOXEL_FACE_SHADING
#define PER_VOXEL_FACE_SHADING 0
#endif

/** Config: find exact hit point to have exact shadows
 */
#ifndef EXACT_SHADOWS
#define EXACT_SHADOWS 1
#endif

/** Config: enable shadows & fog
 */
#ifndef ENABLE_SHADOWS
#define ENABLE_SHADOWS 1
#endif

/** Config: Will run the program without the OpenGL renderer
 */
#ifndef HEADLESS
#define HEADLESS 0
#endif

/** Config: Will use the normal dag instead of the hash dag
 */
#ifndef USE_NORMAL_DAG
#define USE_NORMAL_DAG 0
#endif

/** Config: Will record the tool position, radius and type in the replay
 */
#ifndef RECORD_TOOL_OVERLAY
#define RECORD_TOOL_OVERLAY 0
#endif

/** Config: print edit/copy times
 */
#ifndef VERBOSE_EDIT_TIMES
#define VERBOSE_EDIT_TIMES !USE_VIDEO && !BENCHMARK
#endif

/** Config: count the real number of voxels copied
 */
#ifndef COUNT_COPIED_VOXELS
#define COUNT_COPIED_VOXELS 0
#endif

/** Config: check if a zone is empty and skip it if so in the copy tool
 */
#ifndef COPY_EMPTY_CHECKS
#define COPY_EMPTY_CHECKS !BENCHMARK
#endif

/** Track global new/delete
 */
#ifndef TRACK_GLOBAL_NEWDELETE
#define TRACK_GLOBAL_NEWDELETE 1
#endif

/** Do not decompress the data to copy to a buffer
 */
#ifndef COPY_WITHOUT_DECOMPRESSION
#define COPY_WITHOUT_DECOMPRESSION 1
#endif

/** Apply a transform to the copy
 */
#ifndef COPY_APPLY_TRANSFORM
#define COPY_APPLY_TRANSFORM 1
#endif

/** Allow applying a swirl to the copy
 */
#ifndef COPY_CAN_APPLY_SWIRL 
#define COPY_CAN_APPLY_SWIRL 1
#endif

#if !COPY_WITHOUT_DECOMPRESSION && (COPY_APPLY_TRANSFORM || COPY_CAN_APPLY_SWIRL)
#error "Applying transform/swirl with decompression!"
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Swap the weight bytes for faster read/write
#define CFG_COLOR_SWAP_BYTE_ORDER 1
#define MAX_LEVELS SCENE_DEPTH

#if VERBOSE_EDIT_TIMES
#define EDIT_TIMES(...) __VA_ARGS__
#else
#define EDIT_TIMES(...)
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() __debugbreak()
#endif
#define ASSUME(expr) check(expr); __assume(expr);
#else
#include <csignal>
#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() raise(SIGTRAP)
#endif
#define ASSUME(expr)
#endif

#if defined(_MSC_VER)
#if ENABLE_CHECKS
#undef NDEBUG
#endif
#pragma warning(disable : 4100 4127 4201 4389 4464 4514 4571 4623 4625 4626 4623 4668 4710 4711 4774 4820 5026 5027 5039 5045 )
#define _CRT_SECURE_NO_WARNINGS
#include <intrin.h>
#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64
#endif // ~ _MSC_VER

#if defined(__INTELLISENSE__) || defined(__RSCPP_VERSION) || defined(__CLION_IDE__) || defined(__JETBRAINS_IDE__)
#define __PARSER__
#endif

#ifdef __PARSER__
#error "compiler detected as parser"

#define __CUDACC__
#define __CUDA_ARCH__
#include "crt/math_functions.h"
#include "crt/device_functions.h"
#include "vector_functions.h"
#include <cuda_runtime.h>

#define __host__
#define __device__
#define __global__
#define __device_builtin__

struct TmpIntVector
{
	int x, y, z;
};
TmpIntVector blockIdx;
TmpIntVector blockDim;
TmpIntVector threadIdx;
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cinttypes>
#include <set>
#include <string>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

#if ENABLE_CHECKS
#define checkf(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr, ##__VA_ARGS__); DEBUG_BREAK(); }
#define check(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); }
#define checkAlways(expr) check(expr)
#define checkfAlways(expr, msg, ...) checkf(expr,msg,##__VA_ARGS__)

#define checkEqual(a, b) checkf((a) == (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInf(a, b) checkf((a) < (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInfEqual(a, b) checkf((a) <= (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
__host__ __device__ inline bool __ensure_returns_false(const char* string, int line, const char* expr)
{
	printf(string, line, expr);
#ifndef __CUDA_ARCH__
	static std::set<std::string> set;
	if (set.count(string) == 0)
	{
		set.insert(string);
		DEBUG_BREAK();
	}
#endif
	return false;
}
#define ensure(expr) ((expr) || __ensure_returns_false("Ensure failed " __FILE__ ":%d: %s\n", __LINE__, #expr))
#else
#define checkf(expr, msg, ...)
#define check(...)
#define checkAlways(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); std::abort(); }
#define checkfAlways(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr,##__VA_ARGS__); DEBUG_BREAK(); std::abort(); }
#define checkEqual(a, b)
#define checkInf(a, b)
#define checkInfEqual(a, b)
#define ensure(expr) expr
#endif

#define LOG(msg, ...) printf(msg "\n",##__VA_ARGS__)

#if ENABLE_FORCEINLINE
#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline))
#endif
#else
#define FORCEINLINE
#endif

#if ENABLE_FLATTEN
#ifdef _MSC_VER
#define FLATTEN
#else
#define FLATTEN __attribute__((flatten))
#endif
#else
#define FLATTEN
#endif

#define HOST           __host__ inline FORCEINLINE FLATTEN
#define HOST_RECURSIVE __host__ inline

#define DEVICE           __device__ inline FORCEINLINE FLATTEN
#define DEVICE_RECURSIVE __device__ inline

#define HOST_DEVICE           __host__ __device__ inline FORCEINLINE FLATTEN
#define HOST_DEVICE_RECURSIVE __host__ __device__ inline

using int8 = std::int8_t;
using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using int32 = std::int32_t;
using uint64 = std::uint64_t;
using int64 = std::int64_t;

#if BIG_ENCLOSED_LEAVES
using EnclosedLeavesType = uint64;
#else
using EnclosedLeavesType = uint32;
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

constexpr uint32 imageWidth = 1920;
constexpr uint32 imageHeight = 1080;

constexpr double windowScale = 1;
constexpr uint32 windowWidth = uint32(imageWidth * windowScale);
constexpr uint32 windowHeight = uint32(imageHeight * windowScale);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename T, typename U>
HOST_DEVICE T cast(U value)
{
	static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<int64>::max(), "invalid cast");
	checkf(
		int64(std::numeric_limits<T>::min()) <= int64(value) && int64(value) <= int64(std::numeric_limits<T>::max()),
		"overflow: %" PRIi64 " not in [%" PRIi64 ", %" PRIi64 "]",
		int64(value),
		int64(std::numeric_limits<T>::min()), int64(std::numeric_limits<T>::max()));
	return T(value);
}

template<uint8 Bits, typename Enable = void>
struct TypeForBits;

template<uint8 Bits>
struct TypeForBits<Bits, typename std::enable_if<0 < Bits && Bits <= 8>::type>
{
	typedef uint8 Type;
};
template<uint8 Bits>
struct TypeForBits<Bits, typename std::enable_if<8 < Bits && Bits <= 16>::type>
{
	typedef uint16 Type;
};
template<uint8 Bits>
struct TypeForBits<Bits, typename std::enable_if<16 < Bits && Bits <= 32>::type>
{
	typedef uint32 Type;
};
template<uint8 Bits>
struct TypeForBits<Bits, typename std::enable_if<32 < Bits && Bits <= 64>::type>
{
	typedef uint64 Type;
};


template<uint8 Bits, typename U>
HOST_DEVICE typename TypeForBits<Bits>::Type cast_bits(U value)
{
	static_assert(std::numeric_limits<U>::min() == 0, "invalid cast");
	checkf(
		value < (uint64(1) << Bits),
		"overflow: %" PRIu64 " not in [%" PRIu64 ", %" PRIu64 "]",
		uint64(value),
		uint64(0), (uint64(1) << Bits) - 1);
	return static_cast<typename TypeForBits<Bits>::Type>(value);
}

#define COLOR_BLACK 0x000000
#define COLOR_RED 0xFF0000
#define COLOR_GREEN 0x00FF00
#define COLOR_BLUE 0x0000FF
#define COLOR_YELLOW 0xFFFF00

#ifndef FORCE_ENABLE_TRACY
#define FORCE_ENABLE_TRACY 0
#endif

#if FORCE_ENABLE_TRACY && !defined(TRACY_ENABLE)
#error "FORCE_ENABLE_TRACY = 1 but TRACY_ENABLE undefined"
#endif

#if defined(TRACY_ENABLE) && (!BENCHMARK || FORCE_ENABLE_TRACY)
#include "Tracy.hpp"

#define PROFILE_SCOPEF(Format, ...) \
	ZoneScoped; \
	{ \
		char __String[1024]; \
		const int32 __Size = sprintf(__String, Format, ##__VA_ARGS__); \
		checkInfEqual(__Size, 1024); \
		ZoneName(__String, __Size); \
	}
#define PROFILE_SCOPE(Name) ZoneScopedN(Name)
#define PROFILE_FUNCTION() ZoneScoped

#define ZONE_COLOR(Color) ZoneScopedC(Color)
#define ZONE_METADATA(Format, ...) \
	{ \
		char __String[1024]; \
		const int32 __Size = sprintf(__String, Format, ##__VA_ARGS__); \
		checkInfEqual(__Size, 1024); \
		ZoneText(__String, __Size); \
	}

#define MARK(Name) FrameMarkNamed(Name)
#define MARK_FRAME() FrameMark
#define NAME_THREAD(Name) tracy::SetThreadName(Name)
#define TRACE_ALLOC(Ptr, Size) TracyAlloc(Ptr, Size)
#define TRACE_FREE(Ptr) TracyFree(Ptr)
#else
#define PROFILE_SCOPEF(Format, ...)
#define PROFILE_SCOPE(Name)
#define PROFILE_FUNCTION()
#define ZONE_COLOR(Color)
#define ZONE_METADATA(Format, ...)
#define MARK(Name)
#define MARK_FRAME()
#define NAME_THREAD(Name)
#define TRACE_ALLOC(Ptr, Size)
#define TRACE_FREE(Ptr)
#endif

#if 0
#define PROFILE_FUNCTION_SLOW() PROFILE_FUNCTION()
#define PROFILE_SCOPE_SLOW(Name) PROFILE_SCOPE_SLOW(Name)
#else
#define PROFILE_FUNCTION_SLOW()
#define PROFILE_SCOPE_SLOW(Name)
#endif