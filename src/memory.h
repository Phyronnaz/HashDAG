#pragma once

#include "typedefs.h"
#include <unordered_map>
#include <mutex>
#include <atomic>

enum class EMemoryType
{
    GPU_Managed,
    GPU_Malloc,
    CPU
};

class Memory
{
public:
    template<typename T>
    static T* malloc(const char* name, size_t size, EMemoryType type)
    {
		PROFILE_FUNCTION();
        checkAlways(size % sizeof(T) == 0);
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return reinterpret_cast<T*>(singleton.malloc_impl(size, name, type));
    }
    template<typename T>
    static void free(T* ptr)
    {
		PROFILE_FUNCTION();
        std::lock_guard<std::mutex> guard(singleton.mutex);
        singleton.free_impl(reinterpret_cast<void*>(ptr));
    }
    template<typename T>
    static void realloc(T& ptr, size_t newSize)
    {
		PROFILE_FUNCTION();
        std::lock_guard<std::mutex> guard(singleton.mutex);
		void* copy = reinterpret_cast<void*>(ptr);
        singleton.realloc_impl(copy, newSize);
		ptr = reinterpret_cast<T>(copy);
    }
    template<typename T>
	static void cuda_memcpy(T* dst, const T* src, uint64 Size, cudaMemcpyKind memcpyKind)
    {
		check(Size % sizeof(T) == 0);
		singleton.cuda_memcpy_impl(reinterpret_cast<uint8*>(dst), reinterpret_cast<const uint8*>(src), Size, memcpyKind);
    }

	
    template<typename T>
    static const char* get_alloc_name(const T* ptr)
    {
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return singleton.get_alloc_name_impl(reinterpret_cast<void*>(const_cast<T*>(ptr)));
    }
    static std::string get_stats_string()
    {
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return singleton.get_stats_string_impl();
    }

public:
    inline static size_t get_cpu_allocated_memory()
    {
        return singleton.totalAllocatedCPUMemory;
    }
    inline static size_t get_gpu_allocated_memory()
    {
        return singleton.totalAllocatedGPUMemory;
    }

	inline static size_t get_cxx_cpu_allocated_memory()
	{
		return singleton.cxxCPUMemory.load();
	}

public:
	inline static void track_add_memory( size_t count )
	{
		singleton.cxxCPUMemory += count;
	}
	inline static void track_del_memory( size_t count )
	{
		singleton.cxxCPUMemory -= count;
	}

private:
    Memory() = default;
    ~Memory();

	Memory( Memory const& ) = delete;
	Memory& operator= (Memory const&) = delete;

	void cuda_memcpy_impl(uint8* dst, const uint8* src, uint64 size, cudaMemcpyKind memcpyKind);
    void* malloc_impl(size_t size, const char* name, EMemoryType type);

    void free_impl(void* ptr);
	void realloc_impl(void*& ptr, size_t newSize);
	const char* get_alloc_name_impl(void* ptr) const;
    std::string get_stats_string_impl() const;

    struct Element
    {
        const char* name = nullptr;
        EMemoryType type;
        size_t size = size_t(-1);
    };

    std::mutex mutex;
    size_t totalAllocatedGPUMemory = 0;
    size_t totalAllocatedCPUMemory = 0;
    std::unordered_map<void*, Element> allocations;

	std::atomic<size_t> cxxCPUMemory{0};

    static Memory singleton;
};
