#include "memory.h"
#include "cuda_error_check.h"
#include "utils.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <new>

#if TRACK_GLOBAL_NEWDELETE
namespace
{
	struct CtrlBlock_
	{
		std::size_t blockSize;
	};

	constexpr std::size_t kCtrlBlockSize_ = 32;
}

void* operator new( std::size_t aCount )
{
	Memory::track_add_memory( aCount );

	if( auto* ptr = std::malloc( aCount+kCtrlBlockSize_ ) )
	{
		auto* cb = reinterpret_cast<CtrlBlock_*>(ptr);
		cb->blockSize = aCount;
		return reinterpret_cast<uint8_t*>(ptr)+kCtrlBlockSize_;
	}

	throw std::bad_alloc();
}
void* operator new[]( std::size_t aCount )
{
	Memory::track_add_memory( aCount );

	if( auto* ptr = std::malloc( aCount+kCtrlBlockSize_ ) )
	{
		auto* cb = reinterpret_cast<CtrlBlock_*>(ptr);
		cb->blockSize = aCount;
		return reinterpret_cast<uint8_t*>(ptr)+kCtrlBlockSize_;
	}

	throw std::bad_alloc();
}

void* operator new( std::size_t aCount, std::nothrow_t const& ) noexcept
{
	Memory::track_add_memory( aCount );

	if( auto* ptr = std::malloc( aCount+kCtrlBlockSize_ ) )
	{
		auto* cb = reinterpret_cast<CtrlBlock_*>(ptr);
		cb->blockSize = aCount;
		return reinterpret_cast<uint8_t*>(ptr)+kCtrlBlockSize_;
	}

	return nullptr;
}
void* operator new[]( std::size_t aCount, std::nothrow_t const& ) noexcept
{
	Memory::track_add_memory( aCount );

	if( auto* ptr = std::malloc( aCount+kCtrlBlockSize_ ) )
	{
		auto* cb = reinterpret_cast<CtrlBlock_*>(ptr);
		cb->blockSize = aCount;
		return reinterpret_cast<uint8_t*>(ptr)+kCtrlBlockSize_;
	}

	return nullptr;
}

void operator delete( void* aPtr ) noexcept
{
	if( aPtr )
	{
		auto* base = reinterpret_cast<uint8_t*>(aPtr)-kCtrlBlockSize_;
		auto* cb = reinterpret_cast<CtrlBlock_*>(base);
		
		Memory::track_del_memory( cb->blockSize );
		std::free( base );
	}
}
void operator delete[]( void* aPtr ) noexcept
{
	if( aPtr )
	{
		auto* base = reinterpret_cast<uint8_t*>(aPtr)-kCtrlBlockSize_;
		auto* cb = reinterpret_cast<CtrlBlock_*>(base);
		
		Memory::track_del_memory( cb->blockSize );
		std::free( base );
	}
}

void operator delete( void* aPtr, std::size_t ) noexcept
{
	if( aPtr )
	{
		auto* base = reinterpret_cast<uint8_t*>(aPtr)-kCtrlBlockSize_;
		auto* cb = reinterpret_cast<CtrlBlock_*>(base);
		
		Memory::track_del_memory( cb->blockSize );
		std::free( base );
	}
}
void operator delete[]( void* aPtr, std::size_t ) noexcept
{
	if( aPtr )
	{
		auto* base = reinterpret_cast<uint8_t*>(aPtr)-kCtrlBlockSize_;
		auto* cb = reinterpret_cast<CtrlBlock_*>(base);
		
		Memory::track_del_memory( cb->blockSize );
		std::free( base );
	}
}
#endif // ~ TRACK_GLOBAL_NEWDELETE

Memory Memory::singleton;

inline bool is_gpu_type(EMemoryType type)
{
    return type == EMemoryType::GPU_Malloc || type == EMemoryType::GPU_Managed;
}

inline std::string type_to_string(EMemoryType type)
{
    switch (type)
    {
        case EMemoryType::GPU_Managed:
            return "GPU Managed";
        case EMemoryType::GPU_Malloc:
            return "GPU Malloc";
        case EMemoryType::CPU:
            return "CPU Malloc";
        default:
            check(false);
            return "ERROR";
    }
}

Memory::~Memory()
{
    check(this == &singleton);
    for (auto& it : allocations)
    {
        printf("%s leaked %fMB\n", it.second.name, Utils::to_MB(it.second.size));
    }
    checkAlways(allocations.empty());
    if (allocations.empty())
    {
        printf("No leaks!\n");
    }
}

void Memory::cuda_memcpy_impl(uint8* dst, const uint8* src, uint64 size, cudaMemcpyKind memcpyKind)
{
	const auto BlockCopy = [&]()
	{
		const double Start = Utils::seconds();
		CUDA_CHECKED_CALL cudaMemcpy(dst, src, size, memcpyKind);
		const double End = Utils::seconds();

		return size / double(1u << 30) / (End - Start);
	};

	if (memcpyKind == cudaMemcpyDeviceToDevice)
	{
		PROFILE_SCOPEF("Memcpy HtH %fMB", size / double(1u << 20));
		const double Bandwidth = BlockCopy();
		ZONE_METADATA("%fGB/s", Bandwidth);
	}
	else if (memcpyKind == cudaMemcpyDeviceToHost)
	{
		PROFILE_SCOPEF("Memcpy DtH %fMB", size / double(1u << 20));
		const double Bandwidth = BlockCopy();
		ZONE_METADATA("%fGB/s", Bandwidth);
	}
	else if (memcpyKind == cudaMemcpyHostToDevice)
	{
		PROFILE_SCOPEF("Memcpy HtD %fMB", size / double(1u << 20));
		const double Bandwidth = BlockCopy();
		ZONE_METADATA("%fGB/s", Bandwidth);
	}
}

void* Memory::malloc_impl(size_t size, const char* name, EMemoryType type)
{
	checkAlways(size != 0);
    // printf("Allocating %fMB for %s\n", Utils::to_MB(size), name.c_str());
    void* ptr = nullptr;
    if (is_gpu_type(type))
    {
    	PROFILE_SCOPE("Malloc GPU");
        cudaError_t error;
        if (type == EMemoryType::GPU_Managed)
        {
            error = cudaMallocManaged(&ptr, size);
            if (MEM_ADVISE_READ_MOSTLY)
            {
                // Huge performance improvements when reading
                CUDA_CHECKED_CALL cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, 0);
            }
        }
        else
        {
            check(type == EMemoryType::GPU_Malloc);
            error = cudaMalloc(&ptr, size);
        }
        if (cudaSuccess != error)
        {
            printf("\n\n\n");
            printf("Fatal error when allocating memory!\n");
            std::cout << get_stats_string_impl() << std::endl;
        }
        CUDA_CHECKED_CALL error;
        totalAllocatedGPUMemory += size;
    }
    else
    {
        check(type == EMemoryType::CPU);
    	PROFILE_SCOPE("Malloc CPU");
        ptr = std::malloc(size);
        totalAllocatedCPUMemory += size;
    }
    checkAlways(allocations.find(ptr) == allocations.end());
    allocations[ptr] = { name, type, size };
	TRACE_ALLOC(ptr, size);

    return ptr;
}

void Memory::free_impl(void* ptr)
{
    if (!ptr) return;

    checkAlways(allocations.find(ptr) != allocations.end());
    auto& alloc = allocations[ptr];
    if (is_gpu_type(alloc.type))
    {
    	PROFILE_SCOPE("Free GPU");
        CUDA_CHECKED_CALL cudaFree(ptr);
        totalAllocatedGPUMemory -= alloc.size;
    }
    else
    {
        check(alloc.type == EMemoryType::CPU);
    	PROFILE_SCOPE("Free CPU");
        std::free(ptr);
        totalAllocatedCPUMemory -= alloc.size;
    }
	TRACE_FREE(ptr);
    allocations.erase(ptr);
}

void Memory::realloc_impl(void*& ptr, size_t newSize)
{
	checkAlways(ptr);
    checkAlways(allocations.find(ptr) != allocations.end());
    const auto oldPtr = ptr;
	const auto oldAlloc = allocations[ptr];

    printf("reallocating %s (%s)\n", oldAlloc.name, type_to_string(oldAlloc.type).c_str());

    if (is_gpu_type(oldAlloc.type))
    {
        ptr = malloc_impl(newSize, oldAlloc.name, oldAlloc.type);
        if (ptr) cuda_memcpy_impl(static_cast<uint8*>(ptr), static_cast<uint8*>(oldPtr), oldAlloc.size, cudaMemcpyDefault);
        free_impl(oldPtr);
    }
    else
    {
        check(oldAlloc.type == EMemoryType::CPU);
        allocations.erase(ptr);
        ptr = std::realloc(ptr, newSize);
        allocations[ptr] = { oldAlloc.name, oldAlloc.type, oldAlloc.size };
    }
}

const char* Memory::get_alloc_name_impl(void* ptr) const
{
	checkAlways(ptr);
    checkAlways(allocations.find(ptr) != allocations.end());
    return allocations.at(ptr).name;
}

std::string Memory::get_stats_string_impl() const
{
    struct AllocCount
    {
        size_t size = 0;
        size_t count = 0;

    };
    std::unordered_map<std::string, AllocCount> map;
    for (auto& it : allocations)
    {
        auto name = std::string(it.second.name) + " (" + type_to_string(it.second.type) + ")";
        map[name].size += it.second.size;
        map[name].count++;
    }

    std::vector<std::pair<std::string, AllocCount>> list(map.begin(), map.end());
    std::sort(list.begin(), list.end(), [](auto& a, auto& b) { return a.second.size > b.second.size; });

    std::stringstream ss;
    ss << "GPU memory: " << Utils::to_MB(totalAllocatedGPUMemory) << "MB" << std::endl;
    ss << "CPU memory: " << Utils::to_MB(totalAllocatedCPUMemory) << "MB" << std::endl;
    for (auto& it : list)
    {
        ss << it.first << ": " << Utils::to_MB(it.second.size) << "MB (" << it.second.count << " allocs)" << std::endl;
    }
    return ss.str();
}
